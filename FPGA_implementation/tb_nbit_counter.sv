module tb_nbit_counter;

  // Parameters
  localparam  n = 4;

  //Ports
  reg trig = 0;
  wire logic [n : 0] count;

  nbit_counter # (
                 .n(n)
               )
               nbit_counter_inst (
                 .trig(trig),
                 .count(count)
               );

  always #10 trig = !trig;

  initial
  begin
    #200 $finish;
  end

endmodule
