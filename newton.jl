module newton

    function exptan(init::Number; absvalue::Number=1e-5, iterations::Number=1000)
        x = []
        push!(x, init) # initial guess
        local fc

        for j = 1:iterations # j is the iteration variable

            tmp = x[j] - (exp.(x[j]) - tan(x[j])) / (exp.(x[j]) - sec(x[j])^2)
            push!(x, tmp)
            fc = exp.(tmp) - tan(tmp) # calculate function

            if abs(fc) < absvalue
                break # quit the loop
            end
        end

        x[end], fc # return value of root and function
    end

    print(exptan(-4, absvalue=1e-5))

end  # modul bisection
