function animate_xy_trajectory_ellipse(x, y, theta, slices, system_params)
    % This function plots the trajectory of the ellipse in a channel
    % geometry at each slice index
    L = cast(system_params.L, "double");
    
    yline(L/2, '-', 'Channel Boundary')
    yline(-L/2, '-')
    ylim([-0.75, 0.75])
    xlim([-0.5,5.5])
    daspect([1 1 1])

    h = animatedline('Color','b');

    for n = 1:length(slices)
        clearpoints(h)
        [X_coords, Y_coords] = generate_ellipse_coords(x(1, n), y(1, n), theta(1, n), system_params);
        addpoints(h, X_coords, Y_coords)
        drawnow;
        
%         plot(X_coords, Y_coords)
%         plot(x(n), y(n), '.')
    end
end