import subprocess
import time
import os
import sys
import re
from datetime import datetime

# 检查pandas是否可用，如果不可用则使用csv模块
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    import csv

class AutoTester:
    def __init__(self):
        self.results = []
        self.test_configs = {
            'particle_counts': [100000, 200000, 300000, 400000, 500000, 
                              600000, 700000, 800000, 900000, 1000000],
            'test_name': 'Performance Test',
            'output_file': 'test_results.csv'
        }
        
    def add_encoding_fix(self, filename):
        """确保目标文件包含 import sys，并插入Windows中文编码修复块。"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')

            # 确保存在 import sys（若不存在则在首个import前插入）
            has_import_sys = any(line.strip().startswith('import sys') for line in lines)
            if not has_import_sys:
                # 找到第一个import语句位置，没有则在文件开头插入
                first_import_idx = None
                for i, line in enumerate(lines):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        first_import_idx = i
                        break
                insert_idx = first_import_idx if first_import_idx is not None else 0
                lines.insert(insert_idx, 'import sys')

            modified = False

            # 插入编码修复块（如果不存在）
            if '修复Windows中文编码问题' not in '\n'.join(lines):
                encoding_fix_block = [
                    "# 修复Windows中文编码问题",
                    "if sys.platform.startswith('win'):",
                    "    import codecs",
                    "    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())",
                    "    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())",
                ]

                # 将修复块紧跟在 import sys 之后
                import_sys_idx = None
                for i, line in enumerate(lines):
                    if line.strip().startswith('import sys'):
                        import_sys_idx = i
                        break
                if import_sys_idx is None:
                    import_sys_idx = 0
                for j, fix_line in enumerate(encoding_fix_block):
                    lines.insert(import_sys_idx + 1 + j, fix_line)
                modified = True

            if modified or not has_import_sys:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                print(f"✓ 已修复编码与导入: {filename}")
            else:
                print(f"✓ {filename} 已包含编码修复与 sys 导入")
            return True
        except Exception as e:
            print(f"✗ 添加编码修复到 {filename} 失败: {e}")
            return False
    
    def disable_matplotlib_display(self, filename):
        """禁用matplotlib图片显示"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否已经禁用了显示
            if 'matplotlib.use' not in content:
                # 在matplotlib导入后添加禁用显示的代码
                display_disable = '''
# 禁用matplotlib显示，避免在自动测试时弹出图片
import matplotlib
matplotlib.use('Agg')
'''
                
                # 找到matplotlib导入的位置
                lines = content.split('\n')
                insert_pos = 0
                for i, line in enumerate(lines):
                    if 'import matplotlib' in line or 'from matplotlib' in line:
                        insert_pos = i + 1
                        break
                
                if insert_pos > 0:
                    # 插入禁用显示代码
                    lines.insert(insert_pos, display_disable)
                    modified_content = '\n'.join(lines)
                    
                    # 写回文件
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    
                    print(f"✓ 已禁用 {filename} 的图片显示")
                    return True
                else:
                    print(f"✓ {filename} 未找到matplotlib导入，可能不需要禁用显示")
                    return True
            else:
                print(f"✓ {filename} 已禁用图片显示")
                return True
                
        except Exception as e:
            print(f"✗ 禁用 {filename} 图片显示失败: {e}")
            return False
        
    def modify_dem_main(self, particle_count):
        """修改DEM_main.py中的粒子数量"""
        try:
            with open('DEM_main.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找并替换粒子初始化行
            pattern = r'positions, velocities, radii, mass = initialize_particles\(N=\d+, domain_size=domain_size\)'
            replacement = f'positions, velocities, radii, mass = initialize_particles(N={particle_count}, domain_size=domain_size)'
            
            modified_content = re.sub(pattern, replacement, content)
            
            # 写回文件
            with open('DEM_main.py', 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            print(f"✓ 已修改DEM_main.py: 粒子数量设置为 {particle_count:,}")
            return True
            
        except Exception as e:
            print(f"✗ 修改DEM_main.py失败: {e}")
            return False
    
    def modify_pinn_main(self, particle_count):
        """修改PINN_main_integrated.py中的粒子数量"""
        try:
            with open('PINN_main_integrated.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找并替换粒子初始化行
            pattern = r'positions_np, velocities_np, radii_np, mass_np = initialize_particles\(N=\d+, domain_size=domain_size\)'
            replacement = f'positions_np, velocities_np, radii_np, mass_np = initialize_particles(N={particle_count}, domain_size=domain_size)'
            
            modified_content = re.sub(pattern, replacement, content)
            
            # 写回文件
            with open('PINN_main_integrated.py', 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            print(f"✓ 已修改PINN_main_integrated.py: 粒子数量设置为 {particle_count:,}")
            return True
            
        except Exception as e:
            print(f"✗ 修改PINN_main_integrated.py失败: {e}")
            return False
    
    def run_dem_test(self, particle_count):
        """运行DEM测试"""
        print(f"\n{'='*60}")
        print(f"开始DEM测试 - 粒子数量: {particle_count:,}")
        print(f"{'='*60}")
        
        # 添加编码修复
        if not self.add_encoding_fix('DEM_main.py'):
            return None
        
        # 禁用图片显示
        if not self.disable_matplotlib_display('DEM_main.py'):
            return None
        
        # 修改粒子数量
        if not self.modify_dem_main(particle_count):
            return None
        
        # 运行DEM_main.py
        start_time = time.time()
        try:
            # 设置环境变量解决中文编码问题
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            # 禁用matplotlib显示，避免图片弹出
            env['MPLBACKEND'] = 'Agg'
            
            result = subprocess.run([sys.executable, 'DEM_main.py'], 
                                 capture_output=True, text=True, timeout=1800,
                                 encoding='utf-8', errors='replace', env=env)  # 30分钟超时
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if result.returncode == 0:
                # 从输出中提取计算时间信息
                output_lines = result.stdout.split('\n')
                computation_time = None
                
                for line in output_lines:
                    if '总计算时间:' in line:
                        # 提取时间信息
                        time_match = re.search(r'总计算时间:\s*([\d.]+)\s*s', line)
                        if time_match:
                            computation_time = float(time_match.group(1))
                            break
                
                if computation_time is None:
                    computation_time = total_time
                
                print(f"✓ DEM测试完成 - 总耗时: {total_time:.2f}s, 计算时间: {computation_time:.2f}s")
                
                return {
                    'model': 'DEM',
                    'particle_count': particle_count,
                    'total_time': total_time,
                    'computation_time': computation_time,
                    'status': 'Success'
                }
            else:
                # 即使失败也尝试提取部分信息
                output_lines = result.stdout.split('\n')
                computation_time = None
                
                for line in output_lines:
                    if '总计算时间:' in line:
                        time_match = re.search(r'总计算时间:\s*([\d.]+)\s*s', line)
                        if time_match:
                            computation_time = float(time_match.group(1))
                            break
                
                print(f"✗ DEM测试失败 - 返回码: {result.returncode}")
                print(f"错误输出: {result.stderr}")
                if computation_time:
                    print(f"但成功提取到计算时间: {computation_time:.2f}s")
                
                # 调试：显示前几行输出
                print("调试信息 - 程序输出前10行:")
                for i, line in enumerate(output_lines[:10]):
                    print(f"  {i+1}: {line}")
                
                return {
                    'model': 'DEM',
                    'particle_count': particle_count,
                    'total_time': total_time,
                    'computation_time': computation_time,
                    'status': f'Failed (code: {result.returncode})'
                }
                
        except subprocess.TimeoutExpired:
            print(f"✗ DEM测试超时 (30分钟)")
            return {
                'model': 'DEM',
                'particle_count': particle_count,
                'total_time': 1800,
                'computation_time': None,
                'status': 'Timeout'
            }
        except Exception as e:
            print(f"✗ DEM测试异常: {e}")
            return {
                'model': 'DEM',
                'particle_count': particle_count,
                'total_time': None,
                'computation_time': None,
                'status': f'Exception: {str(e)}'
            }
    
    def run_pinn_test(self, particle_count):
        """运行PINN测试"""
        print(f"\n{'='*60}")
        print(f"开始PINN测试 - 粒子数量: {particle_count:,}")
        print(f"{'='*60}")
        
        # 添加编码修复
        if not self.add_encoding_fix('PINN_main_integrated.py'):
            return None
        
        # 禁用图片显示
        if not self.disable_matplotlib_display('PINN_main_integrated.py'):
            return None
        
        # 修改粒子数量
        if not self.modify_pinn_main(particle_count):
            return None
        
        # 运行PINN_main_integrated.py
        start_time = time.time()
        try:
            # 设置环境变量解决中文编码问题
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            # 禁用matplotlib显示，避免图片弹出
            env['MPLBACKEND'] = 'Agg'
            
            result = subprocess.run([sys.executable, 'PINN_main_integrated.py'], 
                                 capture_output=True, text=True, timeout=1800,
                                 encoding='utf-8', errors='replace', env=env)  # 30分钟超时
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if result.returncode == 0:
                # 从输出中提取计算时间信息
                output_lines = result.stdout.split('\n')
                computation_time = None
                
                for line in output_lines:
                    if '总计算时间:' in line:
                        # 提取时间信息
                        time_match = re.search(r'总计算时间:\s*([\d.]+)\s*s', line)
                        if time_match:
                            computation_time = float(time_match.group(1))
                            break
                
                if computation_time is None:
                    computation_time = total_time
                
                print(f"✓ PINN测试完成 - 总耗时: {total_time:.2f}s, 计算时间: {computation_time:.2f}s")
                
                return {
                    'model': 'PINN',
                    'particle_count': particle_count,
                    'total_time': total_time,
                    'computation_time': computation_time,
                    'status': 'Success'
                }
            else:
                # 即使失败也尝试提取部分信息
                output_lines = result.stdout.split('\n')
                computation_time = None
                
                for line in output_lines:
                    if '总计算时间:' in line:
                        time_match = re.search(r'总计算时间:\s*([\d.]+)\s*s', line)
                        if time_match:
                            computation_time = float(time_match.group(1))
                            break
                
                print(f"✗ PINN测试失败 - 返回码: {result.returncode}")
                print(f"错误输出: {result.stderr}")
                if computation_time:
                    print(f"但成功提取到计算时间: {computation_time:.2f}s")
                
                # 调试：显示前几行输出
                print("调试信息 - 程序输出前10行:")
                for i, line in enumerate(output_lines[:10]):
                    print(f"  {i+1}: {line}")
                
                return {
                    'model': 'PINN',
                    'particle_count': particle_count,
                    'total_time': total_time,
                    'computation_time': computation_time,
                    'status': f'Failed (code: {result.returncode})'
                }
                
        except subprocess.TimeoutExpired:
            print(f"✗ PINN测试超时 (30分钟)")
            return {
                'model': 'PINN',
                'particle_count': particle_count,
                'total_time': 1800,
                'computation_time': None,
                'status': 'Timeout'
            }
        except Exception as e:
            print(f"✗ PINN测试异常: {e}")
            return {
                'model': 'PINN',
                'particle_count': particle_count,
                'total_time': None,
                'computation_time': None,
                'status': f'Exception: {str(e)}'
            }
    
    def save_results(self, show_message=True):
        """保存测试结果到CSV文件"""
        if not self.results:
            if show_message:
                print("没有测试结果可保存")
            return
        
        filename = self.test_configs['output_file']
        
        if HAS_PANDAS:
            # 使用pandas保存
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
        else:
            # 使用csv模块保存
            fieldnames = ['model', 'particle_count', 'total_time', 'computation_time', 'status']
            
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
        
        if show_message:
            print(f"✓ 测试结果已保存到: {filename}")
    
    def print_summary_table(self):
        """打印结果汇总表格"""
        if not self.results:
            print("没有测试结果可显示")
            return
        
        print(f"\n{'='*80}")
        print("测试结果汇总表")
        print(f"{'='*80}")
        
        # 按模型分组显示
        dem_results = [r for r in self.results if r['model'] == 'DEM']
        pinn_results = [r for r in self.results if r['model'] == 'PINN']
        
        if dem_results:
            print("\nDEM模型测试结果:")
            print("-" * 60)
            print(f"{'粒子数量':<12} {'总耗时(s)':<12} {'计算时间(s)':<12} {'状态':<15}")
            print("-" * 60)
            for result in dem_results:
                particle_count = f"{result['particle_count']:,}"
                total_time = f"{result['total_time']:.2f}" if result['total_time'] else "N/A"
                comp_time = f"{result['computation_time']:.2f}" if result['computation_time'] else "N/A"
                status = result['status']
                print(f"{particle_count:<12} {total_time:<12} {comp_time:<12} {status:<15}")
        
        if pinn_results:
            print("\nPINN模型测试结果:")
            print("-" * 60)
            print(f"{'粒子数量':<12} {'总耗时(s)':<12} {'计算时间(s)':<12} {'状态':<15}")
            print("-" * 60)
            for result in pinn_results:
                particle_count = f"{result['particle_count']:,}"
                total_time = f"{result['total_time']:.2f}" if result['total_time'] else "N/A"
                comp_time = f"{result['computation_time']:.2f}" if result['computation_time'] else "N/A"
                status = result['status']
                print(f"{particle_count:<12} {total_time:<12} {comp_time:<12} {status:<15}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("="*80)
        print("自动性能测试开始")
        print("="*80)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"粒子数量范围: {min(self.test_configs['particle_counts']):,} - {max(self.test_configs['particle_counts']):,}")
        print(f"测试模型: DEM + PINN")
        print("="*80)
        
        total_tests = len(self.test_configs['particle_counts']) * 2  # DEM + PINN
        current_test = 0

        # 交替运行：DEM-10万 -> PINN-10万 -> DEM-20万 -> PINN-20万 -> ...
        print(f"\n按交替顺序运行 DEM -> PINN ...")
        for particle_count in self.test_configs['particle_counts']:
            # DEM 测试
            current_test += 1
            print(f"\n进度: {current_test}/{total_tests}")
            dem_result = self.run_dem_test(particle_count)
            if dem_result:
                self.results.append(dem_result)
                # 立即保存结果，防止数据丢失（不显示消息）
                self.save_results(show_message=False)
            time.sleep(2)

            # PINN 测试
            current_test += 1
            print(f"\n进度: {current_test}/{total_tests}")
            pinn_result = self.run_pinn_test(particle_count)
            if pinn_result:
                self.results.append(pinn_result)
                # 立即保存结果，防止数据丢失（不显示消息）
                self.save_results(show_message=False)
            time.sleep(2)
        
        # 显示结果汇总
        self.print_summary_table()
        
        # 保存结果
        self.save_results()
        
        print(f"\n{'='*80}")
        print("所有测试完成!")
        print(f"{'='*80}")

def main():
    """主函数"""
    tester = AutoTester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        if tester.results:
            print("保存已完成的测试结果...")
            tester.print_summary_table()
            tester.save_results()
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        if tester.results:
            print("保存已完成的测试结果...")
            tester.print_summary_table()
            tester.save_results()

if __name__ == "__main__":
    main()
