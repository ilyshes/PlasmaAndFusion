



# convert *.png +append result-sprite.png

using SparseArrays, IterativeSolvers, BandedMatrices, Printf
using Plots, LinearAlgebra


mycommand = `rm -rf  Data/`
run(mycommand)

mycommand = `rm -rf  img/`
run(mycommand)


mycommand = `mkdir  Data`
run(mycommand)

mycommand = `mkdir  img`
run(mycommand)














# Define the grid and parameters
nx = 100  # Number of grid points in x direction
ny = 100
ν = 0.0005
nt = 10000  # Number of time steps
Δx = 1.0/(nx) # Grid spacing in x direction
Δy = 1.0/(ny) # Grid spacing in y direction
Δt = 0.0005 # Time step size



x = [Δx*i for i in 0:nx-1]
y = [Δy*i for i in 0:ny-1]

▧x = [xᵢ for yᵢ in y, xᵢ in x]
▧y = [yᵢ for yᵢ in y, xᵢ in x]

# Initialize the density (ρ), and velocity fields (u, v)
ρ = 1  # density





u = zeros(Float64, ny, nx)   # Velocity in x direction (example values)
υ = zeros(Float64, ny, nx)   # Velocity in x direction (example values)
p = zeros(Float64, ny, nx)   # Velocity in x direction (example values)


# Initialize arrays to store the new values of density
ρ_new = copy(ρ)
u_new = copy(u)  


U_init = 5
u = U_init * sin.(8 * pi * ▧y)
υ = 1 * sin.(2 * pi * ▧x)


u = -sin.(2*pi*▧y)
υ =  sin.(2*pi*▧x*2) 




function  Δⁱʲ()
 
   # Fill the diagonal and off-diagonals with appropriate values
   Is = [1:nx; 1:nx-1; 2:nx; 1; nx]; Js = [1:nx; 2:nx; 1:nx-1; nx; 1]; Vs = [fill(-2,nx); fill(1, 2nx-2);1;1];
   𝛻²x = sparse(Is, Js, Vs)
   
   Is = [1:ny; 1:ny-1; 2:ny; 1; ny]; Js = [1:ny; 2:ny; 1:ny-1; ny; 1]; Vs = [fill(-2,ny); fill(1, 2ny-2);1;1];
   𝛻²y = sparse(Is, Js, Vs)

   𝛻²x = kron(sparse(I,ny,ny), 𝛻²x);
   𝛻²y = kron(𝛻²y, sparse(I,nx,nx));
   
   L = 1/(Δx^2)*𝛻²x + 1/(Δy^2)*𝛻²y 
   return L

end

L = Δⁱʲ()
# Time-stepping loop
for idx in 1:nt
  
    
    global p , u , υ


    
    #plot!(P,ρ[50,:])
    #display(P)

    
    #sleep(0.1)



   

    

  


    ∂²u_∂x² =  (circshift(u, (0,-1)) -  2*u  + circshift(u, (0,1))) / (Δx^2)
    ∂²υ_∂x² =  (circshift(υ, (0,-1)) -  2*υ  + circshift(υ, (0,1))) / (Δx^2)


    ∂²u_∂y² =  (circshift(u, (-1,0)) -  2*u  + circshift(u, (1,0))) / (Δy^2)
    ∂²υ_∂y² =  (circshift(υ, (-1,0)) -  2*υ  + circshift(υ, (1,0))) / (Δy^2)


    global u∂u_∂x = u .* (circshift(u, (0,-1)) -  circshift(u, (0,1))) / (2 * Δx)
    global υ∂u_∂y = υ .* (circshift(u, (-1,0)) -  circshift(u, (1,0))) / (2 * Δy)


    global u∂υ_∂x = u .* (circshift(υ, (0,-1)) -  circshift(υ, (0,1))) / (2 * Δx)
    global υ∂υ_∂y = υ .* (circshift(υ, (-1,0)) -  circshift(υ, (1,0))) / (2 * Δy)

    

    # Update the velocity using the finite difference form of the equation of motion
    u .= u  .+ (- (u∂u_∂x .+ υ∂u_∂y) .+ ( ν * (∂²u_∂x²  .+ ∂²u_∂y²)) ./ ρ) .* Δt 
    υ .= υ  .+ (- (u∂υ_∂x .+ υ∂υ_∂y) .+ ( ν * (∂²υ_∂x²  .+ ∂²υ_∂y²)) ./ ρ) .* Δt          
   

    

    ∂u_∂x =   (circshift(u, (0,-1)) .-  circshift(u, (0,1))) / (2 * Δx)
    ∂υ_dy =   (circshift(υ, (-1,0)) .-  circshift(υ, (1,0))) / (2 * Δy)
    
       
    ∇u⃗ = ∂u_∂x + ∂υ_dy


    rhs = ( ρ  / Δt * ∇u⃗ )
    rhs_col = vec(transpose(rhs))

    @time p = cg(L, rhs_col) # a
   
    #p_general  = L \ rhs_col
    p = transpose(reshape(p,nx,ny))

    


    ∂p∂x_next = (circshift(p, (0,-1)) .-  circshift(p, (0,1))) / (2 * Δx)
    ∂p∂y_next = (circshift(p, (-1,0)) .-  circshift(p, (1,0))) / (2 * Δy)

  
    # Correct the velocities such that the fluid stays incompressible
    u = (u .- Δt / ρ * ∂p∂x_next )
    υ = (υ .- Δt / ρ * ∂p∂y_next )


    # vorticity
    ∂u_∂y =   (circshift(u, (-1,0)) .-  circshift(u, (1,0))) / (2 * Δy)
    ∂υ_∂x =   (circshift(υ, (0,-1)) .-  circshift(υ, (0,1))) / (2 * Δx)
    
    ω = ∂u_∂y .- ∂υ_∂x 

    heatmap(
            x,
            y,
            ω,
            c = :seaborn_icefire_gradient,
            size = (600, 600),
            #clim = (-6, 6),
            axis = true,
            showaxis = true,
            legend = :none,
            title="ω = ∇×u⃗  (vorticity)",
            ticks = true,
            xlabel="X [a.u]", ylabel="Y [a.u]",
            guidefontsize=18,
            titlefontsize=25,
            xtickfont=font(14), 
            ytickfont=font(14),
            margin = 0.0Plots.mm,
            annotations = (
                0.06,
                1.0 - 0.06,
                Plots.text(
                    "$(@sprintf("iter: %05d", idx))",
                    pointsize = 30,
                    color = :white,
                    halign = :left,
                )
            )
        )
        savefig("img/res_$(@sprintf("%05d", idx)).png")


end


mycommand = `ffmpeg -f image2 -framerate 24 -i img/ρ_ρ_ρ_%05d.png animation.gif`
#run(mycommand)