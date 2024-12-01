A = 3000:500:5000;
N_A = length(A);
x=1:N_A;

Krauss = [309.57 452.87 617.26 853.87 1054.25];
CACC = [255.69 305.54 398.45 555.7 854.21];
DDPG = [167.85 205.87 273.48 326.54 377.1];

figure('name','Average travel time against vehicle density'),
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
set(gca,'YTick',[200 300 400 500 600 700 800 900 1000 1100 1200]);
set(gca, 'YLim',[0, 1200]);
xlabel('Traffic Flow (veh/h)');
ylabel('Average Travel Time (s)');