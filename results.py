from collections import defaultdict

EVAL_RESULTS = [
    ('lowdim_blockpush',
        {'loss': {'correlation': 0.003062753710753519, 'negative MMRV': 0.18887997284453492, 'adv over final': -0.09264710317460315, 'smooth correlation': -0.06693657644943729, 'smooth negative MMRV': 0.12003020408163265, 'smooth adv over final': -0.09246594444444445}, 'top_10%_losses': {'correlation': 0.3823423000514194, 'negative MMRV': 0.07959090291921248, 'adv over final': -0.00971551587301589, 'smooth correlation': 0.6934634002888589, 'smooth negative MMRV': 0.02899346938775511, 'smooth adv over final': 0.016881452380952344}, 'top_100_losses': {'correlation': 0.37383986624708543, 'negative MMRV': 0.07651011541072639, 'adv over final': -0.0012169444444444233, 'smooth correlation': 0.6885765057044472, 'smooth negative MMRV': 0.024357959183673457, 'smooth adv over final': 0.022376150793650795}}
        | {'l1': {'correlation': 0.3728649663628342, 'negative MMRV': 0.1327355057705363, 'adv over final': -0.07457043650793646, 'smooth correlation': 0.51464486440019, 'smooth negative MMRV': 0.1139534693877551, 'smooth adv over final': -0.07256235714285714}, 'l2': {'correlation': 0.3289789764982, 'negative MMRV': 0.13335437881873724, 'adv over final': -0.07457043650793646, 'smooth correlation': 0.4507268043994845, 'smooth negative MMRV': 0.1181395918367347, 'smooth adv over final': -0.07256235714285714}, 'l2_sq': {'correlation': 0.003062753710753619, 'negative MMRV': 0.18887997284453492, 'adv over final': -0.09264710317460315, 'smooth correlation': -0.06693657644943722, 'smooth negative MMRV': 0.12003020408163265, 'smooth adv over final': -0.09246594444444445}, 'max': {'correlation': 0.35066775085625096, 'negative MMRV': 0.13250332654446706, 'adv over final': -0.07457043650793646, 'smooth correlation': 0.478872985374902, 'smooth negative MMRV': 0.11077115646258502, 'smooth adv over final': -0.07256235714285714}, 'geom': {'correlation': 0.7032428098179556, 'negative MMRV': 0.08037746096401897, 'adv over final': 0.019384484126984125, 'smooth correlation': 0.8539239523361958, 'smooth negative MMRV': 0.028723673469387753, 'smooth adv over final': 0.02452011904761897}, 'huber': {'correlation': 0.0032462936022387523, 'negative MMRV': 0.1887990495587237, 'adv over final': -0.09264710317460315, 'smooth correlation': -0.06663877603129899, 'smooth negative MMRV': 0.12003020408163265, 'smooth adv over final': -0.09246594444444445}, 'smooth_prob_pi': {'correlation': 0.41598444873434764, 'negative MMRV': 0.13132450780719618, 'adv over final': -0.07457043650793646, 'smooth correlation': 0.622484106594129, 'smooth negative MMRV': 0.09710408163265304, 'smooth adv over final': -0.07256235714285714}, 'smooth_prob_5': {'correlation': 0.47443980643098166, 'negative MMRV': 0.12045023761031906, 'adv over final': -0.06728615079365077, 'smooth correlation': 0.6633525154407746, 'smooth negative MMRV': 0.09509619047619046, 'smooth adv over final': -0.07256235714285714}, 'per_sample_loss': {'correlation': -0.43992942358609166, 'negative MMRV': 0.2609600814663951, 'adv over final': -0.18048265873015867, 'smooth correlation': -0.5371412792098083, 'smooth negative MMRV': 0.22349455782312921, 'smooth adv over final': -0.11703733333333333}}
        | {'closest min': {'correlation': -0.19720305580214947, 'negative MMRV': 0.19609395790902917, 'adv over final': -0.09264710317460315, 'smooth correlation': -0.3177191687448944, 'smooth negative MMRV': 0.12003020408163265, 'smooth adv over final': -0.09246594444444445}, 'closest mean': {'correlation': -0.04228253539064881, 'negative MMRV': 0.19532518669382212, 'adv over final': -0.09264710317460315, 'smooth correlation': -0.13303331750046274, 'smooth negative MMRV': 0.12003020408163265, 'smooth adv over final': -0.09246594444444445}, 'closest max': {'correlation': 0.07808746281683564, 'negative MMRV': 0.18816116768499658, 'adv over final': -0.09264710317460315, 'smooth correlation': 0.04886076936528553, 'smooth negative MMRV': 0.12003020408163265, 'smooth adv over final': -0.09246594444444445}}),
    ('lowdim_tool_hang',
        {'loss': {'correlation': 0.5858481077569022, 'negative MMRV': 0.16213058419243986, 'adv over final': 0.09602777777777771, 'smooth correlation': 0.9066367927509026, 'smooth negative MMRV': 0.062390804597701174, 'smooth adv over final': 0.09819126984126969}, 'top_10%_losses': {'correlation': 0.37849332183411266, 'negative MMRV': 0.16084765177548682, 'adv over final': 0.12071031746031735, 'smooth correlation': 0.8128235116794797, 'smooth negative MMRV': 0.10411494252873564, 'smooth adv over final': 0.09819126984126969}, 'top_100_losses': {'correlation': 0.37999583557180483, 'negative MMRV': 0.17163802978235967, 'adv over final': 0.09664682539682534, 'smooth correlation': 0.8110566237925523, 'smooth negative MMRV': 0.10733333333333335, 'smooth adv over final': 0.04801984126984116}}
        | {'l1': {'correlation': 0.5386750203268622, 'negative MMRV': 0.12412371134020621, 'adv over final': 0.09602777777777771, 'smooth correlation': 0.8903449824495918, 'smooth negative MMRV': 0.05002298850574715, 'smooth adv over final': 0.11430555555555544}, 'l2': {'correlation': 0.534352663849293, 'negative MMRV': 0.2141580756013746, 'adv over final': 0.06439285714285703, 'smooth correlation': 0.8943579055651301, 'smooth negative MMRV': 0.12308045977011496, 'smooth adv over final': 0.04801984126984116}, 'l2_sq': {'correlation': 0.5858481077569024, 'negative MMRV': 0.16213058419243986, 'adv over final': 0.09602777777777771, 'smooth correlation': 0.9066367927509026, 'smooth negative MMRV': 0.062390804597701174, 'smooth adv over final': 0.09819126984126969}, 'max': {'correlation': 0.46055147291689263, 'negative MMRV': 0.47202749140893474, 'adv over final': -0.006718253968253984, 'smooth correlation': 0.8200766463746355, 'smooth negative MMRV': 0.19682758620689653, 'smooth adv over final': -0.021705555555555645}, 'geom': {'correlation': 0.5764711587672297, 'negative MMRV': 0.17956471935853377, 'adv over final': 0.006091269841269842, 'smooth correlation': 0.7711801013844086, 'smooth negative MMRV': 0.11370114942528735, 'smooth adv over final': 0.01730238095238079}, 'huber': {'correlation': 0.5210858948185103, 'negative MMRV': 0.2552119129438717, 'adv over final': 0.06439285714285703, 'smooth correlation': 0.8915853273663806, 'smooth negative MMRV': 0.14774712643678156, 'smooth adv over final': 0.04801984126984116}, 'smooth_prob_pi': {'correlation': 0.6467091849118121, 'negative MMRV': 0.125475372279496, 'adv over final': 0.09664682539682534, 'smooth correlation': 0.8953361161865665, 'smooth negative MMRV': 0.04413793103448277, 'smooth adv over final': 0.04801984126984116}, 'smooth_prob_5': {'correlation': 0.6995497578447901, 'negative MMRV': 0.12437571592210768, 'adv over final': 0.09664682539682534, 'smooth correlation': 0.8901161608960718, 'smooth negative MMRV': 0.0443448275862069, 'smooth adv over final': 0.04801984126984116}, 'per_sample_loss': {'correlation': 0.05789323220500951, 'negative MMRV': 0.5659106529209621, 'adv over final': -0.3069880952380953, 'smooth correlation': 0.053038204566204526, 'smooth negative MMRV': 0.5067126436781609, 'smooth adv over final': -0.13823333333333343}}
        | {'closest min': {'correlation': 0.5236574625052711, 'negative MMRV': 0.28620847651775483, 'adv over final': 0.11594841269841272, 'smooth correlation': 0.8857022158615248, 'smooth negative MMRV': 0.13473563218390805, 'smooth adv over final': 0.11430555555555544}, 'closest mean': {'correlation': 0.5437841010468851, 'negative MMRV': 0.2835967926689576, 'adv over final': 0.12309126984126972, 'smooth correlation': 0.8886216997873635, 'smooth negative MMRV': 0.1453103448275862, 'smooth adv over final': 0.09819126984126969}, 'closest max': {'correlation': 0.5623148410530501, 'negative MMRV': 0.3156701030927834, 'adv over final': 0.06439285714285703, 'smooth correlation': 0.8837234498456171, 'smooth negative MMRV': 0.15956321839080456, 'smooth adv over final': 0.04801984126984116}}),
    ('lowdim_kitchen',
        {'loss': {'correlation': 0.9897603379467317, 'negative MMRV': 0.012045050385299367, 'adv over final': -0.008764739229024987, 'smooth correlation': 0.9910580683632328, 'smooth negative MMRV': 0.005476190476190475, 'smooth adv over final': -0.0035689342403627533}, 'top_10%_losses': {'correlation': 0.988019659866067, 'negative MMRV': 0.01068168346176645, 'adv over final': -0.005116213151927451, 'smooth correlation': 0.9894046340366243, 'smooth negative MMRV': 0.003341269841269807, 'smooth adv over final': 2.3809523809537048e-05}, 'top_100_losses': {'correlation': 0.986937828978804, 'negative MMRV': 0.010677731673582299, 'adv over final': -0.00269671201814059, 'smooth correlation': 0.9861179260040104, 'smooth negative MMRV': 0.0033730158730158424, 'smooth adv over final': 0.0}}
        | {'l1': {'correlation': 0.9881532467201893, 'negative MMRV': 0.011756569847856157, 'adv over final': -0.005116213151927451, 'smooth correlation': 0.9899225740115896, 'smooth negative MMRV': 0.004436507936507929, 'smooth adv over final': -0.0008492063492062663}, 'l2': {'correlation': 0.98701460021883, 'negative MMRV': 0.011495751827702032, 'adv over final': -0.008764739229024987, 'smooth correlation': 0.9890829706161539, 'smooth negative MMRV': 0.003952380952380918, 'smooth adv over final': -0.0008492063492062663}, 'l2_sq': {'correlation': 0.9897603379467317, 'negative MMRV': 0.012045050385299367, 'adv over final': -0.008764739229024987, 'smooth correlation': 0.9910580683632327, 'smooth negative MMRV': 0.005476190476190475, 'smooth adv over final': -0.0035689342403627533}, 'max': {'correlation': 0.9857351619512529, 'negative MMRV': 0.011456233945860499, 'adv over final': -0.005116213151927451, 'smooth correlation': 0.9870441411615607, 'smooth negative MMRV': 0.0037539682539682296, 'smooth adv over final': -0.0007444444444443254}, 'geom': {'correlation': 0.8422258558228233, 'negative MMRV': 0.010673779885398145, 'adv over final': -0.0033588435374150016, 'smooth correlation': 0.9220581834816161, 'smooth negative MMRV': 0.003353174603174571, 'smooth adv over final': 0.0}, 'huber': {'correlation': 0.9912665489150911, 'negative MMRV': 0.011638016202331557, 'adv over final': -0.008764739229024987, 'smooth correlation': 0.9916993844580806, 'smooth negative MMRV': 0.004547619047619032, 'smooth adv over final': -0.0008492063492062663}, 'smooth_prob_pi': {'correlation': 0.7928223643610668, 'negative MMRV': 0.011140090891128234, 'adv over final': -0.0030051020408163875, 'smooth correlation': 0.9066813323047657, 'smooth negative MMRV': 0.003551587301587271, 'smooth adv over final': 2.3809523809537048e-05}, 'smooth_prob_5': {'correlation': 0.6716995232014168, 'negative MMRV': 0.010819996048211817, 'adv over final': -0.0030051020408163875, 'smooth correlation': 0.8278762451970711, 'smooth negative MMRV': 0.003384920634920602, 'smooth adv over final': 2.3809523809537048e-05}, 'per_sample_loss': {'correlation': 0.1209682564929367, 'negative MMRV': 0.13435289468484474, 'adv over final': -0.024551587301587352, 'smooth correlation': -0.2977883980755686, 'smooth negative MMRV': 0.14539285714285705, 'smooth adv over final': -0.049695238095238015}}
        | {'closest min': {'correlation': 0.9892650999175299, 'negative MMRV': 0.011649871566884016, 'adv over final': -0.008764739229024987, 'smooth correlation': 0.99203704304457, 'smooth negative MMRV': 0.004948412698412688, 'smooth adv over final': -0.0015934240362811547}, 'closest mean': {'correlation': 0.9897228542873224, 'negative MMRV': 0.011922544951590609, 'adv over final': -0.008764739229024987, 'smooth correlation': 0.9915846992995657, 'smooth negative MMRV': 0.005222222222222216, 'smooth adv over final': -0.0015934240362811547}, 'closest max': {'correlation': 0.9903349662821226, 'negative MMRV': 0.0120331950207469, 'adv over final': -0.008764739229024987, 'smooth correlation': 0.9910181021234088, 'smooth negative MMRV': 0.005230158730158731, 'smooth adv over final': -0.0008492063492062663}}),
    ('lowdim_pusht',
        {'loss': {'correlation': 0.7386109458897671, 'negative MMRV': 0.02725119001593361}, 'top_10%_losses': {'correlation': 0.7424581865983239, 'negative MMRV': 0.025134571573983466}, 'top_100_losses': {'correlation': 0.7406734040202352, 'negative MMRV': 0.1089832918234525}}
        | {'l1': {'correlation': 0.7548842625246855, 'negative MMRV': 0.02725119001593361}, 'l2': {'correlation': 0.7495254783613362, 'negative MMRV': 0.02725119001593361}, 'l2_sq': {'correlation': 0.7386109458897673, 'negative MMRV': 0.02725119001593361}, 'max': {'correlation': 0.7474944077200927, 'negative MMRV': 0.0273425331779796}, 'geom': {'correlation': 0.8209172811028643, 'negative MMRV': 0.026296711873917147}, 'huber': {'correlation': 0.7495254897259612, 'negative MMRV': 0.02725119001593361}, 'smooth_prob_pi': {'correlation': 0.8623980228656042, 'negative MMRV': 0.04348259453884881}, 'smooth_prob_5': {'correlation': 0.8623977494296868, 'negative MMRV': 0.04348259453884881}}
        | {})
]

EVAL_ROWS = ['per_sample_loss', 'l2_sq', 'top_10%_losses',
             'top_100_losses', 'l1', 'l2', 'l2_sq', 'max',
             'geom', 'huber', 'smooth_prob_pi', 'smooth_prob_5',
             'closest min', 'closest mean', 'closest max']

EVAL_SCORERS = ['correlation', '', 'smooth correlation', '',
                'negative MMRV', '', 'smooth negative MMRV', '',
                'adv over final', '', 'smooth adv over final', '']


def prepare_eval_table():
    for row in EVAL_ROWS:
        for dataset, metrics in EVAL_RESULTS:
            print(f"{row} \t{dataset} \t", end='')
            for scorer in EVAL_SCORERS:
                print(f"{metrics.get(row, {}).get(scorer, '')}\t", end='')
            print()


if __name__ == '__main__':
    prepare_eval_table()