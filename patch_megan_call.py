import re

with open("src/physics/cece_megan.cpp", "r") as f:
    content = f.read()

# In the Run method:
# auto wilt = ResolveImport("wilting_point", import_state);
# double w = has_wilt ? wilt(i, j, 0) : 0.0;
# double gamma_sm = get_gamma_sm(gwet, w);

content = content.replace('auto gwetroot = ResolveImport("soil_moisture_root", import_state);',
                          'auto gwetroot = ResolveImport("soil_moisture_root", import_state);\n    auto wilting_point = ResolveImport("wilting_point", import_state);')

content = content.replace('bool has_gwetroot = (gwetroot.data() != nullptr);',
                          'bool has_gwetroot = (gwetroot.data() != nullptr);\n    bool has_wilt = (wilting_point.data() != nullptr);')

content = content.replace('double gwet = has_gwetroot ? gwetroot(i, j, 0) : 1.0;',
                          'double gwet = has_gwetroot ? gwetroot(i, j, 0) : 1.0;\n            double wilt = has_wilt ? wilting_point(i, j, 0) : 0.0;')

content = content.replace('double gamma_sm = get_gamma_sm(gwet, is_ald2_or_eoh);',
                          'double gamma_sm = get_gamma_sm(gwet, wilt);')

with open("src/physics/cece_megan.cpp", "w") as f:
    f.write(content)
