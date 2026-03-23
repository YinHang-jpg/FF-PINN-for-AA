import pandas as pd
import matplotlib.pyplot as plt
import os

# Read CSV file
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'auto_test_results.csv')
df = pd.read_csv(csv_path)

# Extract data
particles = df['particles'].values
total_collisions = df['total_collisions'].values
elapsed_seconds = df['elapsed_seconds'].values

# Create first plot: Total collisions vs particle count
plt.figure(figsize=(10, 6))
plt.plot(particles, total_collisions, 'o-', linewidth=2, markersize=8, color='#2E86AB')
plt.xlabel('Number of Particles', fontsize=12)
plt.ylabel('Total Collisions', fontsize=12)
plt.title('Total Collisions vs Number of Particles', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('collisions_vs_particles.png', dpi=300, bbox_inches='tight')
print("Saved image: collisions_vs_particles.png")
plt.close()

# Create second plot: Computation time vs particle count
plt.figure(figsize=(10, 6))
plt.plot(particles, elapsed_seconds, 's-', linewidth=2, markersize=8, color='#A23B72')
plt.xlabel('Number of Particles', fontsize=12)
plt.ylabel('Computation Time (seconds)', fontsize=12)
plt.title('Computation Time vs Number of Particles', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('computation_time_vs_particles.png', dpi=300, bbox_inches='tight')
print("Saved image: computation_time_vs_particles.png")
plt.close()

print("All images have been successfully generated!")

