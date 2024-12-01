A = 3000:500:5000;
N_A = length(A);
x=1:N_A;

conservative = [189.54 302.85 482.38 685.24 863.25];
aggressive = [105.85 154.52 285.25 385.45 435.26];

figure('name','Average waiting time against vehicle density'),

plot(A(x),aggressive(x),'-h','MarkerSize',10,'Color',[0.9290 0.6940 0.1250]);
hold on,
plot(A(x),conservative(x),'-vr','MarkerSize',10);

legend(' Aggressive Driver', ' Conservative Driver');
grid on,
figure_FontSize=12;

set(get(gca,'XLabel'),'FontSize',figure_FontSize,'Vertical','top');

set(findobj('FontSize',12),'FontSize',figure_FontSize);
set(findobj(get(gca,'Children'),'LineWidth',0.5),'LineWidth',2.5);
set(gca,'XTick',[3000 3500 4000 4500 5000]);
set(gca, 'XLim',[2800, 5200]);
set(gca,'YTick',[0 200 400 600 800 1000]);
set(gca, 'YLim',[0, 1000]);
xlabel('Traffic Flow (veh/h)');
ylabel('Average Waiting Time (ms)');