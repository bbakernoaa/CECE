import yaml
import sys

def convert_speciation(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    out_config = {"species": {}}

    for mech_spc, sources in data.get('speciation', {}).items():
        layers = []
        for src in sources:
            # Assuming the source field is named something like "MEGAN_{source}"
            # which matches what the MEGAN physics scheme exports if configured right.
            field_name = f"MEGAN_{src['source']}"
            layer = {
                "field": field_name,
                "operation": "add",
                "scale_factor": [src['factor']]
            }
            layers.append(layer)
        out_config["species"][mech_spc] = layers

    out_filename = yaml_file.replace('.yaml', '_cece.yaml')
    with open(out_filename, 'w') as f:
        yaml.dump(out_config, f, sort_keys=False)

    print(f"Generated CECE configuration block in {out_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_megan_speciation_to_cece.py <path_to_yaml>")
        sys.exit(1)
    convert_speciation(sys.argv[1])
