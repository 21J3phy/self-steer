"""Turn a LaTeX-formatted answer into the same answer in plain text (for matched neuron reads)."""
import re


def delatex(s):
    s = s.replace('\\$', '$').replace('\\%', '%')
    for _ in range(3):
        s = re.sub(r'\\(?:d)?frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
        s = re.sub(r'\\(?:text|textbf|mathrm|boxed|mathbf|operatorname)\{([^{}]*)\}', r'\1', s)
    s = s.replace('\\times', 'x').replace('\\cdot', '*').replace('\\div', '/').replace('\\approx', '≈')
    s = s.replace('\\left', '').replace('\\right', '').replace('\\quad', ' ').replace('\\,', ' ')
    s = re.sub(r'\$\$\s*(.*?)\s*\$\$', r'\1', s, flags=re.S)
    s = re.sub(r'\\\[\s*(.*?)\s*\\\]', r'\1', s, flags=re.S)
    s = re.sub(r'\\\((.*?)\\\)', r'\1', s, flags=re.S)
    s = re.sub(r'\$([^$\n]+)\$', r'\1', s)
    s = s.replace('^{', '^(').replace('\\', '')
    s = re.sub(r'\^\(([^)]*)\}', r'^(\1)', s)
    return s
