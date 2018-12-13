module eigenfaces

using Images, ImageSegmentation, ImageMagick, Colors
using Statistics, LinearAlgebra
using Plots

function get_folder(folder::AbstractString, number::Int, resize_y::Int=120, resize_x::Int=80)

    pic_matrix = zeros(Gray{Float64}, number, resize_y, resize_x)
    for i = 1:number
        path = string("eigenvalue_pics/", folder, "/", i, ".jpg")
        img = load(path)
        pic_matrix[i, :, :] = imresize(Gray{Float64}.(img), resize_y, resize_x)
    end

    pic_matrix
end

function get_matrix(folderlist)
    resize_x = 120
    resize_y = 80

    total_num = sum(map(x->x[2], folderlist))[1]
    matrix = zeros(Gray{Float64}, total_num, resize_y*resize_x)
    i = 1
    for f in folderlist
        matrix[i:i+f[2]-1, :] = reshape(get_folder(f[1], f[2], resize_y, resize_x), (f[2], resize_y*resize_x))
        i = i + f[2]
    end

    matrix
end

function normal(A)
    min = findmin(A)[1]
    max = findmax(A)[1]

    (A .- min) / (max .- min)
end

mean(get_folder("clooney", 5), dims=1)[1, :, :] # get a mean face
B = get_matrix([["clooney", 5], ["cooper", 5], ["lawrence", 5], ["stone", 5]]);
C = B' * B # calculate the correlation matrix
E = eigen(C) # get the Eigenvalues and Eigenvectors and save it in E, too play around
Gray.(reshape(normal(E.vectors[:, 9595]), (80, 120))) # look at an eigenface
k = 9580:9600
gr()
plot(k, E.values[k], yaxis=:log) # plot the highest eigenvalues


end  # modul eigenfaces
