from pathlib import Path
import csv
import matplotlib.pyplot as plt



def read_results(csv_file_path: Path):
	"""Read CSV and return three dicts: particle_count -> computation_time for DEM, PINN and COMSOL."""
	dem_times_by_particles = {}
	pinn_times_by_particles = {}
	comsol_times_by_particles = {}
	# Use utf-8-sig to strip BOM if present so headers match
	with csv_file_path.open("r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			if not row:
				continue
			# Normalize keys defensively
			model = (row.get("model") or row.get("\ufeffmodel") or "").strip()
			particle_count_str = (row.get("particle_count") or "").strip()
			computation_time_str = (row.get("computation_time") or "").strip()
			if not model or not particle_count_str or not computation_time_str:
				continue
			try:
				particle_count = int(particle_count_str)
				computation_time = float(computation_time_str)
			except ValueError:
				continue
			if model.upper() == "DEM":
				dem_times_by_particles[particle_count] = computation_time
			elif model.upper() == "PINN":
				pinn_times_by_particles[particle_count] = computation_time
			elif model.upper() == "COMSOL":
				comsol_times_by_particles[particle_count] = computation_time
	return dem_times_by_particles, pinn_times_by_particles, comsol_times_by_particles


def plot_times(dem: dict, pinn: dict, comsol: dict):
	# Prepare series independently to avoid None values
	dem_x = sorted(dem.keys())
	dem_y = [dem[pc] for pc in dem_x]
	pinn_x = sorted(pinn.keys())
	pinn_y = [pinn[pc] for pc in pinn_x]
	comsol_x = sorted(comsol.keys())
	comsol_y = [comsol[pc] for pc in comsol_x]

	plt.figure(figsize=(8, 5))
	plt.plot(dem_x, dem_y, marker="o", linewidth=2, label="DEM")
	plt.plot(pinn_x, pinn_y, marker="s", linewidth=2, label="PINN")
	# Plot COMSOL as points (could be single point at 100000)
	if comsol_x:
		plt.scatter(comsol_x, comsol_y, marker="^", s=60, label="COMSOL", zorder=3)
	plt.title("Computation Time vs Particle Count", fontsize=12)
	plt.xlabel("Particle Count", fontsize=11)
	plt.ylabel("Computation Time (s)", fontsize=11)
	plt.grid(True, linestyle=":", alpha=0.6)
	plt.legend(title="Model")
	plt.tight_layout()

	output_path = Path(__file__).with_name("computation_time_vs_particle_count.png")
	plt.savefig(output_path, dpi=200)
	plt.show()


if __name__ == "__main__":
	csv_path = Path(__file__).with_name("test_results.csv")
	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")
	dem_data, pinn_data, comsol_data = read_results(csv_path)
	if not dem_data and not pinn_data and not comsol_data:
		raise ValueError("No valid data parsed from CSV. Please check the CSV format.")
	plot_times(dem_data, pinn_data, comsol_data)
