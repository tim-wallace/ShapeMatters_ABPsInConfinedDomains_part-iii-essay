function plot_xy_trajectory_ellipse(x, y, theta, slices, system_params)
    % This function plots the trajectory of the ellipse in a channel
    % geometry at each slice index
    L = cast(system_params.L, "double");
    hold on

    for n = slices
        [X_coords, Y_coords] = generate_ellipse_coords(x(n), y(n), theta(n), system_params);
        plot(X_coords, Y_coords)
        plot(x(n), y(n), '.')
    end
%     plot(x,y)
    yline(L/2, '-', 'Channel Boundary')
    yline(-L/2, '-')
    ylim([-0.75, 0.75])
    daspect([1 1 1])

    hold off
end

