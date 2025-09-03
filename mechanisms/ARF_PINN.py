import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# 兼容导入 ARF.py 中的常量与解析函数
try:
    from .ARF import p_0, rho_0, gamma, c_0, frequency, amplitude
except Exception:
    try:
        from mechanisms.ARF import p_0, rho_0, gamma, c_0, frequency, amplitude
    except Exception:
        from ARF import p_0, rho_0, gamma, c_0, frequency, amplitude

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1. 理论数据生成
# -----------------------------
N = 10000
# 生成位置和时间数据
x = (torch.rand(N, 1) * 2.0 - 1.0).to(device)  # 位置范围 [-1, 1] m
t = (torch.rand(N, 1) * 1e-3).to(device)       # 时间范围 [0, 1e-3] s

def theoretical_arf(x, t, particle_radius=1e-6):
    """
    基于 ARF.py 中的物理公式计算理论声辐射力：
    F = π d_p^2 [p(x-√2 d_p/4,t) - p(x+√2 d_p/4,t)] / 4
    其中 p(x,t) = 2π A p0 γ sin(k x) cos(ω t) / λ
        """
    # 计算波长和波数
    wavelength = c_0 / frequency
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * frequency
    
    # 粒子直径
    d_p = 2.0 * particle_radius
    
    # 计算粒子前后两个点的位置
    offset = np.sqrt(2.0) * d_p / 4.0
    x_front = x - offset
    x_back = x + offset
    
    # 使用解析公式计算声压
    p_front = 2.0 * np.pi * amplitude * p_0 * gamma * torch.sin(k * x_front) * torch.cos(omega * t) / wavelength
    p_back = 2.0 * np.pi * amplitude * p_0 * gamma * torch.sin(k * x_back) * torch.cos(omega * t) / wavelength
    
    # 计算压力差和力
    pressure_diff = p_front - p_back
    force_magnitude = np.pi * d_p**2 * pressure_diff / 4.0
    
    return force_magnitude

F_theory = theoretical_arf(x, t).detach()

# -----------------------------
# 2. 特征缩放
# -----------------------------
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1e-3

x_scaled = (x - x_min) / (x_max - x_min)
t_scaled = (t - t_min) / (t_max - t_min)
X_theory = torch.cat((x_scaled, t_scaled), dim=1).to(device)
Y_theory = F_theory  # 直接使用力值，不需要对数变换

# -----------------------------
# 3. 数据准备完成
# -----------------------------

# -----------------------------
# 4. 网络结构
# -----------------------------
class ARFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 128),  # 输入: [x, t]
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)    # 输出: [Fx]
        )

    def forward(self, x):
        return self.layers(x)

model = ARFNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# -----------------------------
# 5. 损失函数定义
# -----------------------------
def loss_function(model):
    """简单的MSE损失：预测力与理论力的差异"""
    pred = model(X_theory)
    return criterion(pred, Y_theory)

# -----------------------------
# 6. 训练循环（仅依据目标损失停止）
# -----------------------------
target_loss = 1e-15
print_interval = 1000

epoch = 0
losses = []

print("开始训练基于ARF.py物理公式的PINN...")
print(f"目标损失: {target_loss:.1e}")

while True:
    model.train()
    optimizer.zero_grad()

    loss = loss_function(model)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % print_interval == 0 or loss.item() < target_loss:
        print(f"Epoch {epoch:05d} | Loss = {loss.item():.3e}")
    
    if loss.item() < target_loss:
        print(f"✅ 达到目标损失 {target_loss:.1e}，在第 {epoch} 个epoch停止训练")
        break
    
    epoch += 1

# 绘制训练曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()

# 测试模型
print("\n=== 模型测试 ===")
model.eval()
with torch.no_grad():
    test_x = torch.linspace(-0.5, 0.5, 10).unsqueeze(1).to(device)
    test_t = torch.zeros(10, 1).to(device)
    test_x_scaled = (test_x - x_min) / (x_max - x_min)
    test_t_scaled = (test_t - t_min) / (t_max - t_min)
    test_input = torch.cat((test_x_scaled, test_t_scaled), dim=1)
    
    pred_forces = model(test_input)
    true_forces = theoretical_arf(test_x, test_t)
    
    print("位置(mm) | 预测力(N) | 理论力(N) | 相对误差")
    print("-" * 50)
    for i in range(10):
        x_val = test_x[i].item() * 1000  # 转换为mm
        pred_val = pred_forces[i].item()
        true_val = true_forces[i].item()
        rel_error = abs(pred_val - true_val) / (abs(true_val) + 1e-12)
        print(f"{x_val:8.1f} | {pred_val:9.2e} | {true_val:9.2e} | {rel_error:8.1%}")

# 保存模型
torch.save(model.state_dict(), 'arf_pinn_model.pth')
print(f"\n模型已保存为 'arf_pinn_model.pth'")

# -----------------------------
# 7. 实时预测函数
# -----------------------------
def predict_arf(x_val, t_val):
    """
    实时预测声辐射力
    参数:
        x_val: 位置 (m)
        t_val: 时间 (s)
    返回:
        Fx: 声辐射力 (N) - 大小由数值决定，方向由正负号决定
    """
    model.eval()
    
    # 转换为张量并归一化
    x_tensor = torch.tensor([[x_val]], dtype=torch.float32, device=device)
    t_tensor = torch.tensor([[t_val]], dtype=torch.float32, device=device)
    
    x_scaled = (x_tensor - x_min) / (x_max - x_min)
    t_scaled = (t_tensor - t_min) / (t_max - t_min)
    input_tensor = torch.cat((x_scaled, t_scaled), dim=1)
    
    with torch.no_grad():
        pred = model(input_tensor)
        force = pred.item()
    
    return force

# 测试实时预测
print("\n=== 实时预测测试 ===")
test_cases = [
    (0.0, 0.0),      # 中心位置，初始时间
    (0.01, 1e-4),    # 偏移位置
    (-0.01, 2e-4),   # 负位置
    (0.005, 5e-4),   # 中间位置
]

for i, (x_test, t_test) in enumerate(test_cases):
    pred_force = predict_arf(x_test, t_test)
    true_force = theoretical_arf(torch.tensor([[x_test]], device=device), 
                                torch.tensor([[t_test]], device=device)).item()
    rel_error = abs(pred_force - true_force) / (abs(true_force) + 1e-12)
    
    print(f"测试 {i+1}: x={x_test*1000:.1f}mm, t={t_test*1000:.1f}ms")
    print(f"  预测力: {pred_force:.2e}N")
    print(f"  理论力: {true_force:.2e}N")
    print(f"  相对误差: {rel_error:.1%}")
    
    # 解释力的方向
    if pred_force > 0:
        direction = "向右（正x方向）"
    else:
        direction = "向左（负x方向）"
    print(f"  力的方向: {direction}")
    print()

print("训练和测试完成！")
