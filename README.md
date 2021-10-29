# Assignment 03

## Speedup Formulas

The speedup of an algorithm is defined as
<img src="https://render.githubusercontent.com/render/math?math=S_p(n) = \frac{T_1(n)}{T_p(n)}">

$S_p(n) = \frac{T_1(n)}{T_p(n)}$.

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

### dna_count.cpp

$T_1(n) = t_a(n)$  
$T_p(n) = t_a(n)/p + t_bN + t_llog_{}p$  
$S_p(n) = \frac{t_a(n)}{t_a(n)/p + t_bN + t_llog_{}p}$

### dna_invert.cpp

$T_1(n) = t_a(n)$  
$T_p(n) = t_a(n)/p + 2(t_bN + t_llog_{}p)$  
$S_p(n) = \frac{t_a(n)}{t_a(n)/p + 2(t_bN + t_llog_{}p)}$

### dna_parse.cpp

$T_1(n) = t_a(n)$  
$T_p(n) = t_a(n)/p + t_bN + t_llog_{}p$  
$S_p(n) = \frac{t_a(n)}{t_a(n)/p + t_bN + t_llog_{}p}$
