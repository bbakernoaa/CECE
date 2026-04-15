with open("src/physics/cece_megan.cpp", "r") as f:
    content = f.read()

# Update get_gamma_sm to actually use the wilting point formula properly
# (and only optionally apply to certain species if needed, though CMAQ seems to pass it in globally in MEGVSA and multiply it out)

content = content.replace("double gwetroot_clamped = std::min(std::max(gwetroot, 0.0), 1.0);", "")

with open("src/physics/cece_megan.cpp", "w") as f:
    f.write(content)
