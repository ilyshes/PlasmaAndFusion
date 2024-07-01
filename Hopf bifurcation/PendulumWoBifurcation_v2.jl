using DifferentialEquations, Plots, Printf
using LinearAlgebra: norm
# Parameters

f = 1.5          # amplitude of the driving force
ω_d = 0.66        # frequency of the driving force
q = 1.1  # different damping coefficients to observe bifurcation


tspan = (0.0, 150.0)  # time span for the simulation

# Define the system of ODEs
function pendulum!(du, u, p, t)
    θ, ω = u
    q, f, ω_d = p
    du[1] = ω
    du[2] = -ω/q - sin(θ) + f*cos(ω_d*t)
end

# Parameters tuple 
p = (q, f, ω_d)



#################   Initial conditions #####################################
θ0₁ = 90  # initial angle
θ0₁ = -θ0₁*pi/180  # initial angle
ω0₁ = -1.1  # initial angular velocity
# Initial conditions
u0 = [θ0₁, ω0₁]  # initial angle and angular velocity

# Solve the ODE
prob = ODEProblem(pendulum!, u0, tspan, p)
sol₁ = solve(prob, abstol=1e-6,reltol=1e-6, Tsit5())

########################################################################

#################   Initial conditions #####################################

θ0₂ = 0  # initial angle
ω0₂ = 0 # initial angular velocity
# Initial conditions
u0 = [θ0₂, ω0₂]  # initial angle and angular velocity

# Solve the ODE
prob = ODEProblem(pendulum!, u0, tspan, p)
sol₂ = solve(prob, abstol=1e-6,reltol=1e-6, Tsit5())

########################################################################





# number of steps
n_iter₁ = size(sol₁, 2)


# Plot the results

# Define the number of frames per second
fps = 5

θ₁  = sol₁[1,:]  
ω₁  = sol₁[2, :]  # initial angular velocity

θ₂  = sol₂[1,:]  
ω₂  = sol₂[2, :]  # initial angular velocity


N_plot_start = 190

anim = @animate for n in N_plot_start:4:n_iter₁ - 10

    p = plot(θ₁[N_plot_start:n], ω₁[N_plot_start:n],  c=:red , 
     linewidth = 4, xlims=(-pi - 0.2, pi + 0.2), 
     ylims=(-3,3),
     ticks = true,
     size = (900, 1000),
     xtickfont=font(20),
     ytickfont=font(20),
     titlefontsize=25,
     guidefontsize=20,
     gridlinewidth=3,
     showaxis = true,
     axis = true,
     legendfontsize=20,
     fg_legend = :false,
     legend_background_color=:transparent,
     legend=:bottomleft,
     label=   @sprintf("θ₀ = %.2f ; θ'₀ = %.2f", θ0₁, ω0₁),
     title="Nonlinear Pendulum Dynamics\n Without Bifurcation",
     ylabel="θ' (angular velocity)",
     xlabel="θ (angle)") 
     # Find the best legend position
     #plt = plot!(legend=find_best_legend_position(p)) 

    # Set the legend position
     
     plot!(θ₂[N_plot_start:n], ω₂[N_plot_start:n],
     label="θ₀ = 0; θ'₀ = 0;  ",
     linewidth = 4,
     c=:blue)

     hline!(p, [0],  linestyle=:dash, color=:black, label = "")
     vline!(p, [0],  linestyle=:dash, color=:black, label = "")


     annotation_text = " dumping q = $q;\n force f = $f;\n diving frequency ω = $ω_d  "
     
     annotate!(-1, 2.7, text(annotation_text, 20, :black))

     scatter!(p, [θ₁[n:n]], [ω₁[n:n]], markersize=20, markercolor=:red, label = "")
     scatter!(p, [θ₂[n:n]], [ω₂[n:n]], markersize=20, markercolor=:blue, label = "")
     
end 

gif(anim, "BeforeBifurcation.gif", fps = fps)