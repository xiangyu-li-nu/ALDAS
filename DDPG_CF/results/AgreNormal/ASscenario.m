A = 3000:500:5000;
N_A = length(A);
x=1:N_A;


aggressive = [21.5 18.85 15.68 13.54 10.82];
conservative = [17.2 15.54 12.87 10.54 8.85]

figure('name','Average speed against vehicle density'),

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
set(gca,'YTick',[6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]);
set(gca, 'YLim',[6, 24]);
xlabel('Traffic Flow (veh/h)');
ylabel('Average Speed (m/s)');