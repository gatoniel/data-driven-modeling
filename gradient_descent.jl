module gradient_descent

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

    function loop(x0::Number, y0::Number; absvalue::Number=1e-6, iterations::Number=1000)
        x = []
        y = []
        push!(x, x0) # initial guess
        push!(y, y0)
        f = []
        push!(f, quadratic_form(x0, y0))

        for j = 1:iterations # j is the iteration variable
            tau = 0.1
            tmpx, tmpy = grad(x[j], y[j], tau)
            push!(x, tmpx)
            push!(y, tmpy)

            push!(f, quadratic_form(x[j+1], y[j+1]))

            if abs(f[j+1] - f[j]) < absvalue
                break # quit the loop
            end
        end

        x[end], y[end], f[end] # return value of root and function
    end

    print(loop(3, 3))

end  # modul gradient_descent
