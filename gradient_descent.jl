module gradient_descent

using Optim

function quadratic_form(x::Number, y::Number)
    (3/2)x^2 + 2x*y + 3y^2 - 2x + 8y
end

function grad(x::Number, y::Number, tau::Number)
    x - tau * (3x + 2y -2), y - tau * (2x + 6y + 8)
end

function tausearch(x::Number, y::Number, tau::Number)
    x0, y0 = grad(x, y, tau)
    quadratic_form(x0, y0)
end

function loop(x0::Number, y0::Number; absvalue::Number=1e-6, iterations::Number=1000, tauinit::Number=0.2)
    x = []
    y = []
    push!(x, x0) # initial guess
    push!(y, y0)
    f = []
    push!(f, quadratic_form(x0, y0))

    for j = 1:iterations # j is the iteration variable
        res = optimize(zeta->tausearch(x[j], y[j], first(zeta)), [tauinit], BFGS())
        tau = Optim.minimizer(res)[1]
        tmpx, tmpy = grad(x[j], y[j], tau)
        push!(x, tmpx)
        push!(y, tmpy)

        push!(f, quadratic_form(x[j+1], y[j+1]))

        print("Step " * string(j) * "\n")
        print("x: " * string(x[j+1]) * "\n")
        print("y: " * string(y[j+1]) * "\n")
        print("f: " * string(f[j+1]) * "\n")
        print("tau: " * string(tau) * "\n")

        if abs(f[j+1] - f[j]) < absvalue
            break # quit the loop
        end
    end

    x[end], y[end], f[end] # return value of root and function
end

print(loop(3, 3))

end  # modul gradient_descent
