with open("src/physics/cece_megan.cpp", "r") as f:
    content = f.read()

# In MEGAN 3, gamma_sm works for isoprene based on wilting point
# It takes `SOILM2` and `wilt` and `d1=0.04`.
# Here's the new logic to insert
new_logic = """double get_gamma_sm(double gwetroot, double wilt) {
    double d1 = 0.04;
    double t1 = wilt + d1;
    double gamma_sm = 1.0;
    if (gwetroot < wilt) {
        gamma_sm = 0.0;
    } else if (gwetroot >= wilt && gwetroot < t1) {
        gamma_sm = (gwetroot - wilt) / d1;
    } else {
        gamma_sm = 1.0;
    }
    return gamma_sm;
}"""

import re
content = re.sub(r'double get_gamma_sm\(double gwetroot, bool is_ald2_or_eoh\) \{.*?return gamma_sm;\n\}', new_logic, content, flags=re.DOTALL)

with open("src/physics/cece_megan.cpp", "w") as f:
    f.write(content)
