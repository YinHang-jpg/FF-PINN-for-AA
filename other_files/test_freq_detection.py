"""Test PINN_FREQ_FOLDER frequency inference."""
import os
import re

test_cases = [
    ('freq_8k', 8000),
    ('freq_10k', 10000),
    ('freq_14k', 14000),
    ('freq_22k', 22000),
]

print("Frequency detection test\n" + "="*60)

for freq_folder, expected_freq in test_cases:
    os.environ['PINN_FREQ_FOLDER'] = freq_folder
    freq_match = re.search(r'freq_(\d+)k', freq_folder)
    if freq_match:
        actual_frequency = int(freq_match.group(1)) * 1000
        status = "ok" if actual_frequency == expected_freq else "XX"
        print(f"{status} {freq_folder} -> {actual_frequency} Hz (expect {expected_freq} Hz)")
    else:
        print(f"XX {freq_folder} -> could not parse")

print("\n" + "="*60)
print("Done.")
