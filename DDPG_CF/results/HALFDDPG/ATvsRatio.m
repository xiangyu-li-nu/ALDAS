Krauss = [1205.6 1005.98 934.65 774.5 660.6];
ACC = [1005.6 845.98 784.65 684.5 598.6];
CACC = [820.25 752.52 682.6 583.2 521.2];
% DDPG = [725.52 654.62 589.5 501.5 453.5];

x = 0:25:100;
y = [Krauss(1) ACC(1) CACC(1); Krauss(2) ACC(2) CACC(2); Krauss(3) ACC(3) CACC(3); Krauss(4) ACC(4) CACC(4); Krauss(5) ACC(5) CACC(5);];
bar(x,y)

legend(' Krauss',' CACC',' DDPG');
grid on,
figure_FontSize=12;

set(gca,'YTick',[0 200 400 600 800 1000 1200]);
set(gca, 'YLim',[0,1300]);
xlabel('AV Ratio (%)');
ylabel('Average Waiting Time (ms)');