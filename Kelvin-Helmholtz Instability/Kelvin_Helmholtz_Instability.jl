



# convert *.png +append result-sprite.png

using SparseArrays, IterativeSolvers, BandedMatrices, Printf
using Plots, LinearAlgebra
using ProgressMeter

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
ny = 200
ν = 0.0002
nt = 5000  # Number of time steps
Δx = 1.0/(nx) # Grid spacing in x direction
Δy = 1.0/(ny) # Grid spacing in y direction
Δt = 0.001 # Time step size



x = [Δx*i for i in 0:nx-1]
y = [Δy*i for i in 0:ny-1]

▧x = [xᵢ for yᵢ in y, xᵢ in x]
▧y = [yᵢ for yᵢ in y, xᵢ in x]

# Initialize fields
u = zeros(Float64, ny, nx)   # Velocity in x direction (example values)
υ = zeros(Float64, ny, nx)   # Velocity in y direction (example values)
p = zeros(Float64, ny, nx)   # Pressure 
s = zeros(Float64, ny, nx)   # υₓ tracing 



ypr = 4 .* ▧y .- 2.0
xpr = ▧x

#  Introducing the υₓ shear layers at y=+1 and y=-1. This is the υₓ profile changing only in the y direction
u =  1/2 .+ 1/2 * (tanh.((ypr .- 1)/0.1) .- tanh.((ypr  .+ 1)/0.1))






#  Introducing the υ_y perturbation at the shear layer. This is the υ_y profile localised at the position of the shear layer.
υ = υ .+ 0.1 * sin.(2*pi .* xpr) .* exp.(-(ypr .- 1).^2/0.01)
υ = υ .+ 0.1 * sin.(2*pi .* xpr) .* exp.(-(ypr .+ 1).^2/0.01)


global s = u



# Laplace operator matrix
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




n = 100           # Update plot every n iterations
# Time-stepping loop

idx_plot = 0


# Time-stepping loop
@showprogress "Timestepping ..."  for idx in 1:1e5
  
    
    global p , u , υ, s



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


    # Solve the Poisson equation for the pressure to respect the incompressibility
    @time p = cg(L, rhs_col) # 
   
    #p_general  = L \ rhs_col
    p = transpose(reshape(p,nx,ny))

    


    ∂p∂x_next = (circshift(p, (0,-1)) .-  circshift(p, (0,1))) / (2 * Δx)
    ∂p∂y_next = (circshift(p, (-1,0)) .-  circshift(p, (1,0))) / (2 * Δy)

  
    # Correct the velocities such that the fluid stays incompressible
    u = (u .- Δt / ρ * ∂p∂x_next )
    υ = (υ .- Δt / ρ * ∂p∂y_next )





    global u∂s_∂x = u .* (circshift(s, (0,-1)) -  circshift(s, (0,1))) / (2 * Δx)
    global υ∂s_∂y = υ .* (circshift(s, (-1,0)) -  circshift(s, (1,0))) / (2 * Δy)

    ∂²s_∂x² =  (circshift(s, (0,-1)) -  2*s  + circshift(s, (0,1))) / (Δx^2)
    ∂²s_∂y² =  (circshift(s, (-1,0)) -  2*s  + circshift(s, (1,0))) / (Δy^2)
    
    s .= s  .+ (- (u∂s_∂x .+ υ∂s_∂y) .+ ( ν * (∂²s_∂x²  .+ ∂²s_∂y²)) ./ ρ) .* Δt 
   



    # vorticity
    ∂u_∂y =   (circshift(u, (-1,0)) .-  circshift(u, (1,0))) / (2 * Δy)
    ∂υ_∂x =   (circshift(υ, (0,-1)) .-  circshift(υ, (0,1))) / (2 * Δx)
    
    ω = ∂u_∂y .- ∂υ_∂x 

   

    # Plot results

    if mod(idx, n) == 0 
        
        global idx_plot += 1
        ps = []
        plot_ω = heatmap(
                x,
                y,
                ω,
                c = :seismic ,
                size = (600, 900),
                #clim = (-12, 12),
                axis = true,
                showaxis = true,
                #legend = true,
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
                        color = :black,
                        halign = :left,
                    )
                )
            )



            plot_p = heatmap(
                x,
                y,
                p,
                c = :seismic ,
                #size = (600, 900),
                clim = (-0.1, 0.1),
                axis = true,
                showaxis = true,
                legend = true,
                title="p  (pressure)",
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
                        color = :black,
                        halign = :left,
                    )
                )
            )
            #push!(plot_p, ps)
            #push!(plot_ω, ps)



            plot_s = heatmap(
                x,
                y,
                s,
                c = :seismic ,
                #size = (600, 900),
                #clim = (-12, 12),
                axis = true,
                showaxis = true,
                legend = true,
                title="υₓ (velocity) shear layer",
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
                        color = :black,
                        halign = :left,
                    )
                )
            )


            plot(
                plot_s, plot_p, plot_ω,
               size=(2600, 1200),
               layout=(1, 3),
               plot_title="Development of the vortex at the shear layer",
               plot_titlefontsize=35, 
               margin=5Plots.mm,
               right_margin=[0Plots.mm 0Plots.mm -10Plots.mm],
               left_margin=[14Plots.mm 0Plots.mm 0Plots.mm],
               bottom_margin= 13Plots.mm,
            )


            # Save figures
            savefig("img/res_$(@sprintf("%05d", idx_plot)).png")
          
    end


end


mycommand = `ffmpeg -f image2 -framerate 24 -i img/ρ_ρ_ρ_%05d.png animation.gif`
#run(mycommand)