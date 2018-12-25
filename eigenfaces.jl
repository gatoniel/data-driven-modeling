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

function get_matrix(folderlist, resize_y::Int=120, resize_x::Int=80)

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

function normal_list!(A)
    for i = 1:size(A)[2]
        A[:, i] = normal(A[:, i])
    end
end

function get_scalars(vectors, A)
    num = size(vectors)[2]

    list = zeros(eltype(vectors[:, 1]), num)
    for i = 1:num
        list[i] = sum(vectors[:, i] .* A)
    end
    list
end

function get_A_from_scalars(scalars, vectors)
    B = zeros(eltype(vectors[:, 1]), size(vectors[:, 1]))
    for i = 1:size(scalars)[1]
        B = B .+ vectors[:, i] .* scalars[i]
    end
    B
end

function plot_scalars(imgs, vectors, k)
    num_imgs = size(imgs)[1]
    num_vecs = size(vectors)[2]
    scalars = zeros(eltype(vectors[:, 1]), num_imgs, num_vecs)
    for i = 1:num_imgs
        scalars[i, :] = get_scalars(vectors, imgs[i])
    end

    display(plot(k, scalars[1, :]))
    for i = 2:num_imgs
        display(plot!(k, scalars[i, :]))
    end

    scalars
end

function compare_img(vectors, imgs, scalars, number::Int, resize_y::Int=120, resize_x::Int=80)
    compare = zeros(eltype(imgs[1]), (resize_y, resize_x*2)) # a bigger picture
    compare[:, 1:resize_x] = get_A_from_scalars(scalars[number, :], vectors) # calculate pic only from k vectors
    compare[:, resize_x+1:end] = imgs[number] # the original
    compare # you dont see much differences
end

function compare_all_imgs(vectors, imgs, scalars, number::Int, resize_y::Int=120, resize_x::Int=80)
    num_imgs = size(imgs)[1]
    compare = zeros(eltype(imgs[1]), (num_imgs*resize_y, resize_x*2)) # a big picture
    for i = 1:num_imgs
        compare[(i-1)*resize_y+1:i*resize_y, :] = compare_img(vectors, imgs, scalars, i, resize_y, resize_x)
    end

    compare
end

resize_x = 80
resize_y = 120

mean(get_folder("clooney", 5), dims=1)[1, :, :] # get a mean face

B = get_matrix([["clooney", 5], ["cooper", 5], ["lawrence", 5], ["stone", 5]], resize_y, resize_x);
C = B' * B # calculate the correlation matrix
E = eigen(C) # get the Eigenvalues and Eigenvectors and save it in E, too play around
Gray.(reshape(normal(E.vectors[:, 9586]), (resize_y, resize_x))) # look at an eigenface

k = 5571:9600 # look only at the highest eigenfaces
gr()

vectors = E.vectors[:, k] # view for highest eigenfaces

V = get_matrix([["von_neumann", 2], ["mcconaughey", 1]], resize_y, resize_x);

imgs = [B[16, :], B[17, :],  B[1, :], V[1, :], V[2, :], V[3, :]] # array of images
scalars = plot_scalars(imgs, vectors, k) # plot and get the scalars of imgs

compare_all_imgs(vectors, imgs, scalars, 6, resize_y, resize_x)

end  # modul eigenfaces
