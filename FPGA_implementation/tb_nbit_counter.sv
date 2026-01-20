module nbit_counter_tb;

  localparam  n = 4;

  reg trig = 0;
  wire logic [n : 0] count;

  nbit_counter # (
    .n(n)
  )
  nbit_counter_inst (
    .trig(trig),
    .count(count)
  );

  always #5 trig = !trig;

  initial begin
    $dumpfile("waveform.vcd");
    $dumpvars(0, nbit_counter_tb);
    #200 $finish;
  end

endmodule