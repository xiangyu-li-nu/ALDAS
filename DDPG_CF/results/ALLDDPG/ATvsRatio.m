Krauss = [1205.6 1005.98 934.65 774.5 660.6];
CACC = [1005.6 845.98 784.65 684.5 598.6];
DDPG = [820.25 752.52 682.6 583.2 521.2];
DDPGALL = [725.52 654.62 589.5 501.5 453.5];

x = 0:25:100;
y = [Krauss(1) CACC(1) DDPG(1) DDPGALL(1); Krauss(2) CACC(2) DDPG(2) DDPGALL(2); Krauss(3) CACC(3) DDPG(3) DDPGALL(3); Krauss(4) CACC(4) DDPG(4) DDPGALL(4); Krauss(5) CACC(5) DDPG(5) DDPGALL(5);];
bar(x,y)

legend(' Krauss', ' CACC', ' DDPG (with human-control)', ' DDPG (all CAV control)');
grid on,
figure_FontSize=12;

set(gca,'YTick',[0 200 400 600 800 1000 1200]);
set(gca, 'YLim',[0,1300]);
xlabel('AV Ratio (%)');
ylabel('Average Waiting Time (ms)');