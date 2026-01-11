module nbit_counter #(
    parameter n = 4
)(
    input logic trig,
    output logic [n : 0] count = 0
);

always @(posedge trig) begin
    count <= count + 1;
end
    
endmodule