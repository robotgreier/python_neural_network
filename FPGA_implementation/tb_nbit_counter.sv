module tb_nbit_counter;

  // Parameters
  localparam  n = 4;

  //Ports
  reg trig = 0;
  wire logic [n-1 : 0] count;

  nbit_counter # (
                 .n(n)
               )
               nbit_counter_inst (
                 .trig(trig),
                 .count(count)
               );

  always #5 trig = !trig;

  initial begin
    // Generate waveform file
    $dumpfile("tb_nbit_counter.vcd");
    $dumpvars(0, tb_nbit_counter);
    
    #320 $finish;
  end

endmodule