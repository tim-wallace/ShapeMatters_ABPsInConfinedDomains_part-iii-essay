function [X_ellipse, Y_ellipse] = generate_ellipse_coords(x, y, theta, system_params)
    % Function to generate the coordinates of the ellipse which can
    % subsequently be plotted
    % (x, y, theta) gives the instantaneous position and orientation of the
    % centre of rotation of the object, and system_params contains 
    % information about the system including its size and eccentricity
    
    a = system_params.a;
    b = system_params.b;
    X_rot = system_params.X_rot;

    s = 0:2*pi/2000:2*pi;

    X = a .* cos(s);
    Y = b .* sin(s);

    X_ellipse = x - X_rot .* cos(theta) + X .* cos(theta) - Y .* sin(theta);
    Y_ellipse = y - X_rot .* sin(theta) + X .* sin(theta) + Y .* cos(theta);
end




