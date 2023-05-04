function plot_xy_trajectory_ellipse_circle(x, y, theta, system_params, N_end)
    % This function plots the trajectory of the ellipse in a channel
    % geometry at each slice index
    R = cast(system_params.R, 'double');
    hold on

    [X_coords, Y_coords] = generate_ellipse_coords(x(1), y(1), theta(1), system_params);
    plot(X_coords, Y_coords)
    plot(x(1), y(1), '+')
    plot(x(1,1:N_end), y(1,1:N_end))
    plot(x(2,1:N_end), y(2,1:N_end))
    plot(x(3,1:N_end), y(3,1:N_end))
    daspect([1 1 1])
    
    s = linspace(0,2*pi,1000);
    plot(R*cos(s), R*sin(s), 'Color', 'black', 'LineStyle','--')

    xlabel('x')
    ylabel('y')

    ylim([-R - 1,R + 1])
    xlim([-R - 1,R + 1])

    hold off
end