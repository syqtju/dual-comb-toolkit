from lib.files import save_configurations

comb_spacings = [(i + 1) * 100e6 for i in range(30)] * 26  # Hz
numbers_of_teeth = [i for i in range(5, 31) for _ in range(30)]  # teeth

nr_configs = min(len(comb_spacings), len(numbers_of_teeth))

print(f"Number of configurations: {nr_configs}")

nr_packs = 16

divisions = nr_configs // nr_packs
if nr_configs % nr_packs != 0:
    divisions += 1

comb_spacings_packs = [
    comb_spacings[i : i + divisions]
    for i in range(0, len(comb_spacings[:nr_configs]), divisions)
]
numbers_of_teeth_packs = [
    numbers_of_teeth[i : i + divisions]
    for i in range(0, len(numbers_of_teeth[:nr_configs]), divisions)
]


for pack in range(nr_packs):
    filename = f"config_pack_{pack + 1}.txt"

    save_configurations(
        filename, comb_spacings_packs[pack], numbers_of_teeth_packs[pack]
    )

print(f"Configurations per pack: {divisions}.")
print("Configuration packs generated successfully.")
