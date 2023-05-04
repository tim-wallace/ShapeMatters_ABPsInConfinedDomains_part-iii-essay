function plot_xy_trajectory_ellipse_square(x, y, theta, system_params)
    % This function plots the trajectory of the ellipse in a channel
    % geometry at each slice index
    L = cast(system_params.L, 'double');
    hold on

    [X_coords, Y_coords] = generate_ellipse_coords(x(1), y(1), theta(1), system_params);
    plot(X_coords, Y_coords)
    plot(x(1), y(1), '+')
    plot(x(1,1:10001), y(1,1:10001))
    plot(x(2,1:10001), y(2,1:10001))
    plot(x(3,1:10001), y(3,1:10001))
    yline(L/2, '-')
    yline(-L/2, '-')
    xline(L/2, '-')
    xline(-L/2, '-')
    daspect([1 1 1])
    xlabel('x')
    ylabel('y')

    ylim([-L/2 - 1,L/2 + 1])
    xlim([-L/2 - 1,L/2 + 1])

    hold off
end

