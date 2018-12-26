module nonlinear

using LinearAlgebra, NLsolve

jacobian(x::Number, y::Number) = [2+3*x^2 1; 1+y+exp(x) x]

func(x::Number, y::Number) = [2*x+y+x^3; x+x*y+exp(x)]

function f!(F, x) # used for nlsolve
    F[1] = func(x[1], x[2])[1]
    F[2] = func(x[1], x[2])[2]
end

function loop(x0; absvalue::Number=1e-6, iterations::Number=1000)
    x = x0

    for j = 1:iterations # j is the iteration variable
        J = jacobian(x[1], x[2])
        f = func(x[1], x[2])

        print("Step " * string(j) * "\n")
        print("x: " * string(x) * "\n")
        print("f: " * string(f) * "\n")
        print("J: " * string(J) * "\n")

        if norm(f) < absvalue
            break # quit the loop
        end

        df = -J \ f
        x = x + df
    end

    x, f(x[1], x[2]) # return value of root and function
end

x0 = [0.; 0.]
loop(x0)

res = nlsolve(f!, x0)
print("With nlsolve\n")
print(res.zero)

end  # modul nonlinear
