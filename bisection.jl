module bisection

    function exptan(xl::Number, xr::Number; absvalue::Number=1e-5, iterations::Number=1000)
        local xc
        local fc
        for j = 1:iterations # j cuts the interval

            xc = (xl + xr) / 2;     # calculate the midpoint
            fc = exp.(xc) - tan(xc) # calculate function

            if fc > 0
                xl = xc; # move left boundary
            else
                xr = xc; # move right boundary
            end

            if abs(fc) < absvalue
                break # quit the loop
            end
        end

        xc, fc # return value of root and function
    end

    using Plots # load Plots
    plotly() # use plotly as backend
    x=-4:0.1:3
    y=exp.(x) - tan.(x)
    plot(x, y, show=true)

    print(exptan(-4, -2.8, absvalue=1e-5))

end  # modul bisection
