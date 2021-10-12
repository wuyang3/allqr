clear; close all; clc;

figure(1);
set(gcf, 'color', [1 1 1]);
xlim([-10 120]);
ylim([-10 40]);
grid on;
hold on;

[x, y] = ginput;
dist = 0.0;
for i = 2:numel(x)
    x1 = x(i-1);
    x2 = x(i);
    y1 = y(i-1);
    y2 = y(i);
    dist = dist + sqrt((x2-x1)^2 + (y2-y1)^2);
    plot(x, y, 'o');
end

axis equal;

fid = fopen('ref_line_seg_values.txt','w');
fprintf(fid, ['reference_lines_x {' '\n']);
for i = 1:numel(x)
    fprintf(fid, ['\tvalue: ', num2str(x(i))  '\n']);
end
fprintf(fid, ['}' '\n']);

fprintf(fid, ['reference_lines_y {' '\n']);
for i = 1:numel(x)
    fprintf(fid, ['\tvalue: ', num2str(y(i))  '\n']);
end
fprintf(fid, ['}' '\n']);

fclose(fid);

