# Compile
xvlog --sv nbit_counter.sv nbit_counter_tb.sv
# Elaborate  
xelab -debug typical nbit_counter_tb -s sim
# Run
xsim sim -runall
