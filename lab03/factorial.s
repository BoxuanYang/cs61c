.globl factorial

.data
n: .word 8

.text
main:
    la t0, n
    lw a0, 0(t0) # a0 := 8
    jal ra, factorial

    addi a1, a0, 0 # a1 := a0, return register
    addi a0, x0, 1 # a0 := 1
    ecall # Print Result

    addi a1, x0, '\n' # a1 := '\n'
    addi a0, x0, 11 # a0 := 11
    ecall # Print newline

    addi a0, x0, 10 # a0 := 10
    ecall # Exit

factorial:
    # YOUR CODE HERE
    addi t0, x0, 1
loop:    
    beq a0, x0, Quit # a0 := n
    
    mul t0, t0, a0 
    
    
    addi a0, a0, -1
    j loop
    
Quit:
    mv a0, t0
    jr ra



    
    