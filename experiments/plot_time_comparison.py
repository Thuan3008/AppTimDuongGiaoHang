"""Read saved timed runs, validate coverage, export PNG/SVG, CSV and HTML."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path,
                        default=Path('results/time_comparison_n25_120s_seeds0_29'))
    args = parser.parse_args()
    root = args.directory.resolve()
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    seconds = protocol['seconds']
    names = protocol['algorithms']
    rows = []
    for name in names:
        seeds = protocol['seeds'] if name in ['TD-ERAPO-Time', 'GA', 'GA-LS', 'ARO-Original', 'POA-Original'] else [None]
        for seed in seeds:
            data = json.loads((root / f'{name}_seed{seed if seed is not None else "det"}.json').read_text(encoding='utf-8'))
            if not (data['algorithm'] == name and data['seed'] == seed and data['status'] == 'completed'
                    and data['feasible'] and data['time_limit_seconds'] == seconds):
                raise ValueError('Invalid run identity/protocol')
            history = data['history']
            if any(b['seconds'] < a['seconds'] or b['duration'] > a['duration'] + 1e-6
                   for a, b in zip(history, history[1:])):
                raise ValueError('Non-monotonic incumbent history')
            within = [h['duration'] for h in history if h['seconds'] <= seconds]
            if not within:
                raise ValueError('No feasible incumbent within deadline; do not silently omit this run')
            strict = min(within)
            if not math.isclose(strict, data['best_duration_within_deadline'], abs_tol=1e-6):
                raise ValueError('Deadline value disagrees with incumbent history')
            bks = data['duration'] / (1 + data['gap_to_bks_percent'] / 100)
            data.update(strict_duration=strict, strict_gap=100 * (strict / bks - 1), bks=bks,
                        measured_runtime=data.get('search_runtime_seconds', data.get('runtime_seconds')))
            rows.append(data)
    if max(r['bks'] for r in rows) - min(r['bks'] for r in rows) > 1e-6:
        raise ValueError('Runs disagree on BKS')
    out = root / 'visualizations'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    colors = {n: plt.get_cmap('tab10')(i) for i, n in enumerate(names)}
    grouped = {n: [r for r in rows if r['algorithm'] == n] for n in names}
    table = []
    for n, group in grouped.items():
        values = [r['strict_duration'] for r in group]
        gaps = [r['strict_gap'] for r in group]
        table.append({'algorithm': n, 'runs': len(group), 'best_duration_120s': min(values),
                      'mean_duration_120s': stats.mean(values), 'sd_duration': stats.stdev(values) if len(values)>1 else None,
                      'mean_gap_120s_percent': stats.mean(gaps), 'sd_gap_percentage_points': stats.stdev(gaps) if len(gaps)>1 else None,
                      'mean_reported_final_duration': stats.mean(r['duration'] for r in group),
                      'runs_improved_after_deadline': sum(r['duration'] < r['strict_duration'] - 1e-6 for r in group),
                      'mean_runtime_seconds': stats.mean(r['measured_runtime'] for r in group),
                      'mean_nfe': stats.mean(r['nfe'] for r in group) if group[0]['nfe'] is not None else None,
                      'termination': ', '.join(sorted({r['termination'] for r in group}))})
    table.sort(key=lambda x: x['mean_gap_120s_percent'])
    with (out / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    (out / 'metrics.json').write_text(json.dumps(table,indent=2),encoding='utf-8')
    images=[]
    def save(fig, filename):
        fig.tight_layout()
        for suffix in ['png','svg']:
            fig.savefig(out / f'{filename}.{suffix}', dpi=180, bbox_inches='tight')
        plt.close(fig)
        images.append(filename+'.png')
    fig, ax=plt.subplots(figsize=(12,6))
    order=[r['algorithm'] for r in table]
    means=[r['mean_gap_120s_percent'] for r in table]
    ax.barh(order,means,color=[colors[n] for n in order],xerr=[r['sd_gap_percentage_points'] or 0 for r in table],capsize=3)
    for i,r in enumerate(table):
        ax.text(means[i]+(r['sd_gap_percentage_points'] or 0)+.15,i,f'{means[i]:.3f}%  (n={r["runs"]})',va='center',fontsize=10)
    ax.invert_yaxis();ax.set_xlim(0,max(means)*1.35)
    ax.set_xlabel('Gap đến BKS (%) - thấp hơn tốt hơn')
    ax.set_title(f'Hong Kong HYB n25 BPR-heavy | Nghiệm hoàn tất trong {seconds:g} giây\nThanh: Mean Gap; vạch: SD qua seed; phương pháp tất định: một lượt')
    ax.grid(axis='x',alpha=.2);save(fig,'01_gap_ranking')
    stochastic=[n for n in order if len(grouped[n])>1]
    fig,ax=plt.subplots(figsize=(12,6))
    ax.boxplot([[r['strict_gap'] for r in grouped[n]] for n in stochastic],tick_labels=stochastic,showmeans=True)
    for i,n in enumerate(stochastic,1):
        offsets=np.linspace(-.15,.15,len(grouped[n]))
        ax.scatter(i+offsets,[r['strict_gap'] for r in grouped[n]],color=colors[n],s=14,alpha=.7)
    ax.set_ylabel('Gap đến BKS (%)');ax.set_title('Phân bố kết quả trong hạn - mỗi chấm là một seed')
    ax.grid(axis='y',alpha=.2);save(fig,'02_gap_distribution')
    fig,ax=plt.subplots(figsize=(12,6))
    grid=np.linspace(0,seconds,481)
    for name in names:
        trajectories=[]
        for r in grouped[name]:
            h=[h for h in r['history'] if h['seconds']<=seconds]
            t=np.array([x['seconds'] for x in h]);v=np.array([100*(x['duration']/r['bks']-1) for x in h])
            idx=np.searchsorted(t,grid,side='right')-1
            trajectory=np.full(len(grid),np.nan);mask=idx>=0;trajectory[mask]=v[idx[mask]]
            trajectories.append(trajectory)
        values=np.array(trajectories);available=np.isfinite(values).all(axis=0)
        curve=np.full(len(grid),np.nan);curve[available]=values[:,available].mean(axis=0)
        ax.step(grid,curve,where='post',label=name,color=colors[name],linestyle='-' if len(grouped[name])>1 else '--')
    ax.set(xlabel='Thời gian tìm kiếm thực tế (giây)',ylabel='Gap đến BKS (%)',xlim=(0,seconds),
           title='Hội tụ theo thời gian - giữ nghiệm tốt nhất đã hoàn tất\nĐường trung bình chỉ vẽ từ khi đủ seed có nghiệm; không chuẩn hóa chỉ số lịch sử')
    ax.legend(fontsize=9,ncol=3);ax.grid(alpha=.2);save(fig,'03_convergence_time')
    fig,ax=plt.subplots(figsize=(12,6))
    ax.barh(order,[next(r['mean_runtime_seconds'] for r in table if r['algorithm']==n) for n in order],color=[colors[n] for n in order])
    ax.axvline(seconds,color='black',ls='--',label=f'Ngân sách {seconds:g}s');ax.invert_yaxis()
    ax.set_xlabel('Thời gian ghi nhận (giây)');ax.set_title('Thời gian thực tế - giới hạn mềm; phạm vi đo theo từng bộ chạy')
    ax.legend();save(fig,'04_runtime')
    fig,ax=plt.subplots(figsize=(11,5))
    ax.bar(stochastic,[stats.mean(r['nfe'] for r in grouped[n]) for n in stochastic],color=[colors[n] for n in stochastic])
    ax.set_ylabel('NFE trung bình');ax.set_title('Số nghiệm đánh giá thực tế của nhóm ngẫu nhiên\nKhông quy đổi số lần đánh giá tuyến của heuristic thành NFE')
    ax.grid(axis='y',alpha=.2);save(fig,'05_nfe')
    fig,ax=plt.subplots(figsize=(12,6))
    fitness=[r['mean_duration_120s'] for r in table]
    ax.barh(order,fitness,color=[colors[n] for n in order],
            xerr=[r['sd_duration'] or 0 for r in table],capsize=3,
            label='Trung bình (nhóm tất định: một lượt); vạch: SD')
    ax.scatter([r['best_duration_120s'] for r in table],range(len(table)),
               marker='D',color='black',s=30,zorder=4,label='Giá trị tốt nhất')
    for i,r in enumerate(table):
        ax.text(fitness[i]+(r['sd_duration'] or 0)+180,i,
                f'{fitness[i]:,.2f}  (n={r["runs"]})',va='center',fontsize=10)
    ax.invert_yaxis();ax.set_xlim(0,max(fitness)*1.24)
    ax.set_xlabel('Fitness = tổng Duration với nghiệm khả thi (giây) - thấp hơn tốt hơn')
    ax.set_title(f'So sánh Fitness trong {seconds:g} giây | Hong Kong HYB n25 BPR-heavy')
    ax.grid(axis='x',alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),fontsize=9)
    save(fig,'06_fitness_comparison')
    after=sum(r['runs_improved_after_deadline'] for r in table)
    note=f'Đủ {len(rows)} lượt hợp lệ. Có {after} lượt cải thiện sau hạn; hình và bảng chính dùng nghiệm hoàn tất trong {seconds:g} giây. Nhóm ngẫu nhiên: 30 seed; phương pháp tất định: 1 lượt, không tính SD giả. NFE không phải ngân sách dừng của đợt này. Thời gian nạp dữ liệu được loại khỏi đồng hồ tìm kiếm; xử lý cuối có thể làm runtime vượt hạn. Các phương pháp có phạm vi đo xử lý cuối khác nhau, không diễn giải chênh lệch nhỏ của runtime là ưu thế thuật toán.'
    heads=['Thuật toán','Số lượt','Best Duration','Mean Duration','SD','Mean Gap (%)','Giây ghi nhận']
    body=''
    for r in table:
        cells=[r['algorithm'],r['runs'],f'{r["best_duration_120s"]:,.2f}',f'{r["mean_duration_120s"]:,.2f}',f'{r["sd_duration"]:,.2f}' if r['sd_duration'] is not None else '—',f'{r["mean_gap_120s_percent"]:.3f}',f'{r["mean_runtime_seconds"]:.2f}']
        body+='<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in cells)+'</tr>'
    page='<!doctype html><meta charset="utf-8"><title>Đối chứng thời gian n25</title><style>body{font:16px Arial;max-width:1200px;margin:35px auto;padding:0 20px;background:#f3f5f8;color:#172538}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}img{width:100%;margin:20px 0;background:white}p{line-height:1.6}</style><h1>Đối chứng thời gian - Hong Kong HYB n25 BPR-heavy</h1><p>'+html.escape(note)+'</p><table><tr>'+''.join('<th>'+s+'</th>' for s in heads)+'</tr>'+body+'</table>'+''.join(f'<img src="{im}" alt="{im}">' for im in images)
    (out/'index.html').write_text(page,encoding='utf-8')
    (out/'README.txt').write_text(note+'\nNguồn: protocol.json và 154 JSON từng lượt. Không đọc checkpoint trùng hoặc tạo thêm số liệu.\nTái tạo: .\\.venv\\Scripts\\python.exe -m experiments.plot_time_comparison\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'runs':len(rows),'after_deadline_improvements':after,'ranking':[(r['algorithm'],r['mean_gap_120s_percent']) for r in table]},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
