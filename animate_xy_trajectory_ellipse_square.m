function animate_xy_trajectory_ellipse_square(x, y, theta, slices, system_params)
    % This function plots the trajectory of the ellipse in a channel
    % geometry at each slice index
    L = cast(system_params.L, "double");
    
    yline(L/2, '-')
    yline(-L/2, '-')
    xline(L/2, '-')
    xline(-L/2, '-')
    ylim([1.1*-L/2, 1.1*L/2])
    xlim([1.1*-L/2, 1.1*L/2])
    daspect([1 1 1])
    xlabel('x')
    ylabel('y')


    h = animatedline('Color','b');

    F(length(slices)) = struct('cdata',[],'colormap',[]);

    for n = 1:length(slices)
        clearpoints(h)
        [X_coords, Y_coords] = generate_ellipse_coords(x(1, slices(n)), y(1, slices(n)), theta(1, slices(n)), system_params);
        addpoints(h, X_coords, Y_coords)
        drawnow;
        cdata = print('-RGBImage','-r150');
        F(n) = im2frame(cdata);

    end

    video = VideoWriter('test_video2.mp4', 'MPEG-4');
    video.FrameRate = 500;
    open(video)
    writeVideo(video, F)
    close(video)
end