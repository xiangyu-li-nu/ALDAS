A = 3000:500:5000;
N_A = length(A);
x=1:N_A;

Krauss = [389.57 728.87 1117.26 1490.87 1954.25];
CACC = [245.69 368.54 632.58 852.28 1224.28];
DDPG = [134.98 210.21 340.14 456.42 552.15];

figure('name','Average waiting time against vehicle density'),
plot(A(x),Krauss(x),'-+k','MarkerSize',10);
hold on,
plot(A(x),CACC(x),'-h','MarkerSize',10,'Color',[0.9290 0.6940 0.1250]);
hold on,
plot(A(x),DDPG(x),'-vr','MarkerSize',10);

legend(' Krauss', ' CACC', ' DDPG');
grid on,
figure_FontSize=12;

set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');

set(findobj('FontSize',12),'FontSize',figure_FontSize);
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2.5);
set(gca,'XTick',[3000 3500 4000 4500 5000]);
set(gca, 'XLim',[2800, 5200]);
set(gca,'YTick',[0 200 400 600 800 1000 1200 1400 1600 1800 2000]);
set(gca, 'YLim',[0, 2000]);
xlabel('Traffic Flow (veh/h)');
ylabel('Average Waiting Time (ms)');