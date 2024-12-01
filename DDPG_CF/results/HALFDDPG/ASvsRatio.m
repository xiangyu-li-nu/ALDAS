Krauss = [4.02 5.15 5.98 6.82 8.05];
CACC = [4.56 5.84 6.68 7.51 8.78];
DDPG = [8.54 10.54 11.87 13.64 15.28];
% DDPG = [10.55 12.65 14.98 16.75 17.28];

x = 0:25:100;
y = [Krauss(1) CACC(1) DDPG(1); Krauss(2) CACC(2) DDPG(2); Krauss(3) CACC(3) DDPG(3); Krauss(4) CACC(4) DDPG(4); Krauss(5) CACC(5) DDPG(5);];
bar(x,y)

legend(' Krauss',' CACC',' DDPG');
grid on,
figure_FontSize=12;

set(gca,'YTick',[0 2 4 6 8 10 12 14 16 18]);
set(gca, 'YLim',[0,18]);
xlabel('AV Ratio (%)');
ylabel('Average Speed (m/s)');