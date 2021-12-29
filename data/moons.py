import numpy as np
K = 2
M = 600
N = 2
A = np.array(
	[
	[	1.791601	,	-0.232715	],
	[	0.909865	,	0.074057	],
	[	0.9863	,	-0.529365	],
	[	1.321273	,	-0.333136	],
	[	0.35774	,	-0.315487	],
	[	0.168261	,	1.047554	],
	[	-0.270111	,	0.885529	],
	[	1.767937	,	-0.174601	],
	[	0.426011	,	-0.301393	],
	[	1.02188	,	0.141551	],
	[	0.649199	,	-0.438387	],
	[	-1.002498	,	0.206058	],
	[	1.144269	,	-0.476669	],
	[	-0.417917	,	0.960166	],
	[	-0.319555	,	0.967214	],
	[	0.759042	,	-0.531708	],
	[	2.021865	,	0.252025	],
	[	1.881863	,	0.131983	],
	[	1.960274	,	0.253909	],
	[	0.954478	,	-0.40588	],
	[	0.812119	,	0.536938	],
	[	-0.664905	,	0.569331	],
	[	1.010732	,	0.146917	],
	[	-0.864119	,	0.280467	],
	[	-0.082557	,	1.115292	],
	[	-0.038462	,	0.976484	],
	[	0.466657	,	0.976025	],
	[	1.099788	,	-0.57772	],
	[	1.384908	,	-0.439556	],
	[	0.504707	,	0.787774	],
	[	1.526108	,	-0.393883	],
	[	1.02589	,	0.028329	],
	[	-0.01411	,	0.037782	],
	[	0.1406	,	0.155274	],
	[	0.61321	,	-0.262021	],
	[	-1.021392	,	0.158323	],
	[	0.730535	,	0.774611	],
	[	1.93456	,	0.350378	],
	[	0.56255	,	0.907112	],
	[	1.452693	,	-0.356088	],
	[	-0.489082	,	0.885821	],
	[	0.988498	,	0.068214	],
	[	1.316744	,	-0.471954	],
	[	0.062941	,	0.943566	],
	[	0.10096	,	0.055016	],
	[	-0.67112	,	0.729739	],
	[	0.841088	,	0.348822	],
	[	-1.028587	,	0.263329	],
	[	-0.113602	,	0.450221	],
	[	1.151373	,	-0.539225	],
	[	-0.751959	,	0.955005	],
	[	-0.895929	,	0.484063	],
	[	-0.591622	,	0.691714	],
	[	1.804762	,	0.567586	],
	[	0.469577	,	0.97235	],
	[	-0.393829	,	1.077013	],
	[	-0.951681	,	-0.038212	],
	[	0.310878	,	0.87794	],
	[	-1.030546	,	0.371785	],
	[	-0.939129	,	0.218995	],
	[	-0.527601	,	0.663402	],
	[	-0.247517	,	0.894894	],
	[	-0.005519	,	0.101468	],
	[	-0.123555	,	0.970297	],
	[	-0.267702	,	0.967377	],
	[	-0.701856	,	0.625573	],
	[	0.720043	,	0.812154	],
	[	1.406736	,	-0.471659	],
	[	1.662827	,	-0.156303	],
	[	1.452566	,	-0.330738	],
	[	-0.68367	,	0.772744	],
	[	0.036911	,	0.302164	],
	[	-0.142419	,	0.953604	],
	[	-0.981774	,	0.28634	],
	[	0.285932	,	-0.343877	],
	[	0.364227	,	1.069835	],
	[	0.535142	,	0.893736	],
	[	1.875217	,	0.456322	],
	[	0.738495	,	0.733329	],
	[	0.382055	,	0.906607	],
	[	0.034391	,	0.08969	],
	[	-0.519006	,	0.953398	],
	[	-0.936105	,	0.591664	],
	[	-0.491593	,	1.019262	],
	[	0.901916	,	-0.439845	],
	[	0.608468	,	0.789192	],
	[	-0.096073	,	0.519764	],
	[	0.340662	,	-0.232993	],
	[	-0.341142	,	0.898763	],
	[	-0.957519	,	0.444743	],
	[	0.824192	,	-0.394245	],
	[	0.083195	,	0.329319	],
	[	0.715537	,	0.523084	],
	[	1.921849	,	0.273704	],
	[	0.411432	,	0.804667	],
	[	1.188252	,	-0.480303	],
	[	-0.479103	,	1.023572	],
	[	0.549501	,	0.862885	],
	[	1.097065	,	-0.382608	],
	[	0.907856	,	0.278374	],
	[	2.03524	,	0.419671	],
	[	1.053429	,	-0.493118	],
	[	0.637179	,	0.701461	],
	[	0.401667	,	-0.309676	],
	[	-0.245064	,	0.957019	],
	[	0.609012	,	0.732759	],
	[	1.875937	,	-0.133308	],
	[	0.407118	,	-0.063538	],
	[	-1.031081	,	0.07351	],
	[	0.494787	,	0.994217	],
	[	0.858812	,	-0.522606	],
	[	1.109202	,	-0.345203	],
	[	0.990752	,	0.174731	],
	[	1.905993	,	-0.156002	],
	[	-0.571407	,	0.627375	],
	[	2.044001	,	0.065157	],
	[	1.881721	,	0.333952	],
	[	-0.103804	,	0.401941	],
	[	-0.741731	,	0.535912	],
	[	-0.977995	,	0.514074	],
	[	1.050796	,	0.466869	],
	[	1.308513	,	-0.427822	],
	[	-0.086991	,	0.559661	],
	[	0.349592	,	-0.334119	],
	[	0.064643	,	0.019023	],
	[	0.071793	,	0.036695	],
	[	1.369682	,	-0.446498	],
	[	0.09122	,	0.075622	],
	[	1.887143	,	0.011587	],
	[	-0.205755	,	0.987688	],
	[	1.910805	,	0.255238	],
	[	-0.941797	,	0.249509	],
	[	0.945393	,	0.426937	],
	[	-0.967905	,	0.308739	],
	[	1.021014	,	0.33741	],
	[	1.674048	,	-0.171484	],
	[	0.515648	,	0.677117	],
	[	0.100853	,	1.067876	],
	[	-0.59475	,	0.659799	],
	[	0.940449	,	0.151816	],
	[	-0.348462	,	0.907656	],
	[	1.714678	,	-0.184941	],
	[	2.008483	,	0.56775	],
	[	0.128553	,	0.20537	],
	[	2.116283	,	0.205136	],
	[	0.133547	,	1.11239	],
	[	0.954317	,	0.280439	],
	[	0.501639	,	-0.332085	],
	[	1.187165	,	-0.384014	],
	[	1.925749	,	0.09642	],
	[	0.657503	,	0.934056	],
	[	0.013502	,	0.928529	],
	[	0.65034	,	-0.413786	],
	[	-0.984719	,	0.183536	],
	[	0.795382	,	0.389716	],
	[	0.707179	,	0.689951	],
	[	-0.11638	,	0.80479	],
	[	1.092514	,	-0.473588	],
	[	1.063419	,	-0.496128	],
	[	-1.105109	,	0.152076	],
	[	0.049361	,	0.027374	],
	[	1.388295	,	-0.350581	],
	[	1.896719	,	0.068841	],
	[	1.305581	,	-0.42411	],
	[	0.994181	,	0.429445	],
	[	-0.378252	,	0.891862	],
	[	-1.017357	,	0.290151	],
	[	2.004626	,	-0.079248	],
	[	-0.887245	,	0.596095	],
	[	0.66405	,	-0.490071	],
	[	1.059422	,	0.183544	],
	[	1.74046	,	-0.235707	],
	[	0.167685	,	-0.110212	],
	[	1.451352	,	-0.566233	],
	[	-0.709154	,	0.575462	],
	[	0.72283	,	-0.364963	],
	[	-0.872305	,	0.330628	],
	[	0.878657	,	-0.434822	],
	[	0.355648	,	0.922248	],
	[	1.670476	,	-0.272585	],
	[	0.073584	,	1.016561	],
	[	1.037495	,	-0.474585	],
	[	0.468434	,	-0.248223	],
	[	0.126273	,	-0.119361	],
	[	-0.017076	,	0.224389	],
	[	0.554707	,	0.799759	],
	[	1.248891	,	-0.547739	],
	[	-0.603665	,	0.61875	],
	[	0.875309	,	0.48583	],
	[	-1.147547	,	0.11707	],
	[	0.800367	,	0.612028	],
	[	0.590051	,	0.856833	],
	[	0.250508	,	-0.227384	],
	[	0.905763	,	0.41158	],
	[	0.615618	,	-0.103443	],
	[	0.282727	,	-0.076801	],
	[	2.072339	,	0.233816	],
	[	-1.01429	,	0.157928	],
	[	-0.74142	,	0.591137	],
	[	-0.856415	,	0.479286	],
	[	0.046333	,	0.191529	],
	[	-0.91865	,	0.160515	],
	[	0.885879	,	0.495831	],
	[	0.604186	,	-0.559469	],
	[	0.021663	,	0.9229	],
	[	-0.637931	,	0.583815	],
	[	0.915372	,	-0.440183	],
	[	0.621716	,	0.634771	],
	[	1.260921	,	-0.560883	],
	[	1.014083	,	-0.499545	],
	[	0.151652	,	0.194227	],
	[	1.993913	,	0.296771	],
	[	-1.017462	,	0.179831	],
	[	0.477188	,	-0.368549	],
	[	-0.277735	,	1.110345	],
	[	0.835357	,	0.52942	],
	[	0.102214	,	0.191658	],
	[	1.572697	,	-0.540154	],
	[	1.478924	,	-0.475871	],
	[	0.281877	,	-0.289916	],
	[	1.025611	,	0.245358	],
	[	1.860827	,	0.006241	],
	[	0.514759	,	0.821319	],
	[	0.273817	,	-0.320867	],
	[	0.901455	,	0.225483	],
	[	-1.069281	,	0.400784	],
	[	-0.030084	,	0.498753	],
	[	-0.671307	,	0.879277	],
	[	1.960473	,	0.280644	],
	[	1.504494	,	-0.255745	],
	[	0.264981	,	0.050398	],
	[	1.22364	,	-0.512314	],
	[	-1.134766	,	0.278623	],
	[	-0.974195	,	0.212517	],
	[	0.443444	,	-0.158982	],
	[	1.807212	,	0.047552	],
	[	1.743547	,	-0.156743	],
	[	-0.468461	,	0.989141	],
	[	-0.953973	,	0.220889	],
	[	-0.887108	,	0.462942	],
	[	1.612051	,	-0.443681	],
	[	1.838956	,	-0.082654	],
	[	2.074665	,	0.484677	],
	[	-0.852034	,	0.593267	],
	[	0.894798	,	0.57809	],
	[	0.10329	,	-0.100845	],
	[	-0.018242	,	0.387745	],
	[	1.219084	,	-0.416951	],
	[	1.885794	,	0.166954	],
	[	-0.045306	,	0.262857	],
	[	0.384182	,	-0.312942	],
	[	0.424828	,	-0.323466	],
	[	-0.836374	,	0.536566	],
	[	0.600848	,	-0.39095	],
	[	0.595843	,	0.780814	],
	[	1.552861	,	-0.160973	],
	[	0.648158	,	0.720945	],
	[	-0.112563	,	0.96388	],
	[	1.019631	,	0.310229	],
	[	0.801083	,	0.517925	],
	[	-0.96438	,	0.599436	],
	[	1.772713	,	-0.178397	],
	[	-0.534316	,	0.874034	],
	[	1.116894	,	-0.473125	],
	[	1.553817	,	-0.25254	],
	[	0.58616	,	0.887725	],
	[	2.042621	,	0.565388	],
	[	-0.774845	,	0.611728	],
	[	1.168443	,	-0.327639	],
	[	0.742014	,	0.60732	],
	[	-0.957451	,	0.153177	],
	[	0.248476	,	0.978591	],
	[	0.067899	,	-0.028948	],
	[	0.766653	,	-0.448604	],
	[	1.086829	,	-0.351451	],
	[	0.896319	,	0.142106	],
	[	1.698899	,	-0.111714	],
	[	0.942007	,	-0.674093	],
	[	1.363193	,	-0.310067	],
	[	-0.01883	,	0.239546	],
	[	0.362418	,	1.071954	],
	[	0.247872	,	0.844397	],
	[	-0.867983	,	0.231103	],
	[	1.400039	,	-0.327729	],
	[	0.369464	,	0.896005	],
	[	0.977905	,	0.00628	],
	[	0.021139	,	0.199578	],
	[	0.281075	,	0.831091	],
	[	1.80933	,	-0.209192	],
	[	0.239362	,	-0.097621	],
	[	0.648507	,	0.591588	],
	[	-0.056864	,	1.039425	],
	[	0.899575	,	0.575521	],
	[	-0.783013	,	0.680031	],
	[	0.203356	,	1.133624	],
	[	-0.659646	,	0.79922	],
	[	-0.075387	,	0.286821	],
	[	2.090361	,	0.451976	],
	[	0.164784	,	0.00893	],
	[	1.042931	,	-0.673549	],
	[	0.725369	,	0.57579	],
	[	0.540225	,	0.951174	],
	[	2.060997	,	0.110957	],
	[	1.70584	,	-0.100538	],
	[	-0.544973	,	0.78865	],
	[	0.617176	,	-0.486671	],
	[	1.802764	,	-0.058505	],
	[	0.381359	,	-0.032657	],
	[	1.696751	,	-0.289192	],
	[	1.904539	,	-0.262949	],
	[	0.866195	,	-0.385134	],
	[	0.954448	,	-0.507595	],
	[	0.795069	,	0.639242	],
	[	-0.153568	,	1.025322	],
	[	0.920786	,	-0.441278	],
	[	0.671429	,	0.729887	],
	[	1.534411	,	-0.421303	],
	[	1.583593	,	-0.454251	],
	[	0.997446	,	-0.002362	],
	[	-1.035027	,	0.134306	],
	[	-1.152512	,	0.228584	],
	[	-0.228339	,	1.083285	],
	[	1.804272	,	0.042299	],
	[	-0.159494	,	0.892063	],
	[	-0.484476	,	0.856255	],
	[	0.307621	,	0.898479	],
	[	0.801969	,	-0.467562	],
	[	0.408404	,	-0.302595	],
	[	-0.886395	,	0.259238	],
	[	0.467144	,	-0.421314	],
	[	-0.119973	,	1.001118	],
	[	-0.103943	,	1.081248	],
	[	0.667335	,	-0.427213	],
	[	0.00335	,	0.112666	],
	[	0.223367	,	0.377171	],
	[	0.586464	,	0.775774	],
	[	0.460639	,	0.736234	],
	[	0.082399	,	1.001458	],
	[	0.918172	,	-0.484734	],
	[	1.962939	,	-0.066525	],
	[	0.118731	,	0.06892	],
	[	0.571249	,	-0.411506	],
	[	0.069577	,	0.351739	],
	[	-1.110611	,	0.301289	],
	[	2.014948	,	0.026608	],
	[	0.081127	,	-0.136257	],
	[	0.930787	,	-0.461409	],
	[	1.004617	,	0.350988	],
	[	-0.680907	,	0.389967	],
	[	0.841563	,	-0.50932	],
	[	0.834751	,	0.30481	],
	[	-0.754818	,	0.586588	],
	[	0.048706	,	0.108006	],
	[	-0.591859	,	0.811789	],
	[	0.09952	,	0.979432	],
	[	0.638668	,	0.837775	],
	[	0.10872	,	0.297778	],
	[	0.969978	,	-0.615904	],
	[	0.080203	,	0.240975	],
	[	1.867709	,	0.050742	],
	[	0.290583	,	1.069232	],
	[	1.028871	,	0.434365	],
	[	0.997745	,	-0.112365	],
	[	0.492449	,	-0.425091	],
	[	-0.418427	,	1.14171	],
	[	0.637487	,	0.597894	],
	[	0.332492	,	-0.293346	],
	[	1.344612	,	-0.380353	],
	[	0.6811	,	-0.398364	],
	[	0.355149	,	0.681106	],
	[	-0.897767	,	0.412621	],
	[	1.467454	,	-0.364429	],
	[	0.467106	,	0.905684	],
	[	1.845551	,	-0.136347	],
	[	0.489614	,	-0.351203	],
	[	-0.400236	,	0.999937	],
	[	-0.85818	,	0.672241	],
	[	-0.715056	,	0.47333	],
	[	0.061311	,	0.999065	],
	[	-0.827218	,	0.632291	],
	[	-0.213242	,	1.075975	],
	[	-1.090597	,	0.048675	],
	[	1.413777	,	-0.317036	],
	[	1.999001	,	0.26204	],
	[	0.241936	,	-0.152012	],
	[	0.97811	,	0.427104	],
	[	0.124915	,	0.086359	],
	[	-0.696297	,	0.581184	],
	[	1.077852	,	0.234741	],
	[	0.498924	,	0.774133	],
	[	-0.322624	,	0.936848	],
	[	1.989965	,	0.224794	],
	[	-0.908386	,	0.289675	],
	[	-0.317901	,	0.884029	],
	[	1.833193	,	-0.159062	],
	[	1.904838	,	0.140416	],
	[	0.841814	,	0.473738	],
	[	1.824484	,	0.243228	],
	[	-0.368973	,	0.878394	],
	[	0.565007	,	-0.498081	],
	[	1.039248	,	0.281098	],
	[	-0.097494	,	0.812773	],
	[	-1.108019	,	0.443997	],
	[	-0.611828	,	0.80114	],
	[	1.175029	,	-0.492529	],
	[	0.480626	,	0.836912	],
	[	0.486915	,	-0.166564	],
	[	-0.13727	,	0.47737	],
	[	-1.029036	,	0.146296	],
	[	1.691031	,	-0.19115	],
	[	1.750606	,	-0.15192	],
	[	1.164699	,	-0.476945	],
	[	1.030391	,	-0.391056	],
	[	2.106356	,	0.268029	],
	[	0.204588	,	0.565479	],
	[	0.877491	,	0.442597	],
	[	0.117908	,	0.949495	],
	[	1.037556	,	0.003905	],
	[	-0.777153	,	0.486948	],
	[	-0.677022	,	0.675537	],
	[	-0.604345	,	0.720681	],
	[	0.821516	,	0.651344	],
	[	0.70678	,	0.56441	],
	[	1.68719	,	-0.292142	],
	[	0.293294	,	0.875527	],
	[	0.886765	,	0.084158	],
	[	-0.016118	,	0.100201	],
	[	1.99246	,	0.388546	],
	[	0.592809	,	0.78928	],
	[	-0.945185	,	0.292729	],
	[	1.048518	,	0.356514	],
	[	0.913472	,	0.365468	],
	[	0.104169	,	0.553585	],
	[	-0.012673	,	0.84125	],
	[	0.704663	,	0.663682	],
	[	0.073286	,	-0.048571	],
	[	2.082896	,	-0.013488	],
	[	1.375995	,	-0.387495	],
	[	0.050247	,	0.88421	],
	[	1.370979	,	-0.379131	],
	[	0.596843	,	-0.508268	],
	[	0.360501	,	0.931962	],
	[	-0.021264	,	1.176475	],
	[	1.927542	,	0.324289	],
	[	0.261924	,	-0.218549	],
	[	2.066671	,	0.541995	],
	[	0.39939	,	-0.516133	],
	[	1.5279	,	-0.411361	],
	[	1.726889	,	-0.302399	],
	[	0.148811	,	0.083955	],
	[	0.977409	,	-0.015154	],
	[	-0.958303	,	0.219835	],
	[	0.892517	,	-0.645732	],
	[	0.444118	,	0.73481	],
	[	1.672386	,	-0.334045	],
	[	-0.706613	,	0.664575	],
	[	2.083851	,	0.201364	],
	[	-0.266735	,	0.970011	],
	[	0.756063	,	0.669032	],
	[	0.931305	,	0.160225	],
	[	1.288355	,	-0.603587	],
	[	0.033783	,	0.4565	],
	[	0.290642	,	-0.228801	],
	[	-0.661822	,	0.661081	],
	[	0.813624	,	0.316076	],
	[	-0.545642	,	0.738484	],
	[	-1.11281	,	0.069158	],
	[	-0.809279	,	0.658295	],
	[	-0.139281	,	0.887464	],
	[	1.389282	,	-0.550723	],
	[	2.17944	,	0.424662	],
	[	0.395928	,	-0.371297	],
	[	0.200865	,	-0.226815	],
	[	-0.017826	,	0.855131	],
	[	1.309779	,	-0.685471	],
	[	-0.155653	,	0.304927	],
	[	0.406744	,	0.923231	],
	[	1.923366	,	0.321087	],
	[	1.745484	,	0.045911	],
	[	0.068232	,	0.95675	],
	[	1.942174	,	0.211734	],
	[	-1.062679	,	0.127526	],
	[	1.298338	,	-0.44943	],
	[	-0.709808	,	0.701525	],
	[	-0.445791	,	0.735646	],
	[	0.976236	,	0.232807	],
	[	0.853503	,	0.637287	],
	[	0.793372	,	-0.53967	],
	[	0.09781	,	0.282502	],
	[	1.388039	,	-0.45489	],
	[	0.26302	,	-0.25217	],
	[	1.257335	,	-0.554203	],
	[	0.968322	,	0.142561	],
	[	-1.02774	,	0.377529	],
	[	1.98557	,	0.419814	],
	[	0.661266	,	0.484373	],
	[	1.627477	,	-0.187019	],
	[	-0.853784	,	0.55876	],
	[	0.984109	,	0.479019	],
	[	1.162092	,	-0.473672	],
	[	0.876461	,	0.621127	],
	[	1.77946	,	-0.142854	],
	[	1.061375	,	0.074222	],
	[	0.75781	,	0.736628	],
	[	0.938775	,	0.418559	],
	[	1.566348	,	-0.299458	],
	[	-0.08681	,	0.956384	],
	[	1.982289	,	0.578627	],
	[	0.466332	,	-0.274082	],
	[	0.326148	,	-0.239071	],
	[	0.937944	,	-0.621045	],
	[	-0.440394	,	0.913127	],
	[	0.69969	,	0.721561	],
	[	0.236368	,	0.34111	],
	[	1.550314	,	-0.400957	],
	[	1.918873	,	0.298386	],
	[	0.065625	,	0.017805	],
	[	-0.265751	,	0.864803	],
	[	-0.043478	,	-0.066615	],
	[	-0.757573	,	0.692437	],
	[	1.934405	,	0.022955	],
	[	0.808054	,	0.526203	],
	[	0.707106	,	-0.66619	],
	[	0.325635	,	0.865812	],
	[	-0.155886	,	0.333444	],
	[	1.6131	,	-0.150216	],
	[	0.444947	,	-0.242887	],
	[	-0.122402	,	0.91572	],
	[	-0.255269	,	0.78747	],
	[	0.582977	,	0.677471	],
	[	2.077525	,	0.252581	],
	[	0.893657	,	0.712069	],
	[	0.24063	,	0.177709	],
	[	2.022443	,	0.127277	],
	[	1.463428	,	-0.280939	],
	[	0.927964	,	-0.465195	],
	[	1.304835	,	-0.338075	],
	[	2.049286	,	0.239123	],
	[	1.027429	,	-0.463901	],
	[	0.382978	,	-0.203464	],
	[	0.138504	,	0.174534	],
	[	1.822413	,	0.211857	],
	[	0.25081	,	-0.192487	],
	[	0.279542	,	0.865359	],
	[	0.141422	,	0.807369	],
	[	0.396891	,	0.920344	],
	[	0.332893	,	0.895112	],
	[	0.216207	,	0.974421	],
	[	0.602703	,	-0.342903	],
	[	-0.350218	,	0.976823	],
	[	0.202396	,	1.099348	],
	[	0.496265	,	0.956125	],
	[	-0.446832	,	0.827102	],
	[	-0.905248	,	0.350952	],
	[	1.894295	,	0.002903	],
	[	-0.691527	,	0.733984	],
	[	-0.823913	,	0.717182	],
	[	-1.066774	,	0.059935	],
	[	1.442619	,	-0.254901	],
	[	1.118947	,	-0.434665	],
	[	0.837395	,	0.626418	],
	[	0.445191	,	-0.588969	],
	[	0.738017	,	-0.458299	],
	[	2.020174	,	0.343508	],
	[	0.146129	,	1.020186	],
	[	0.018503	,	-0.069201	],
	[	0.265806	,	0.896399	],
	[	-0.309466	,	0.883391	],
	[	-0.627442	,	0.908502	],
	[	-0.70369	,	0.810984	],
	[	1.459002	,	-0.440659	],
	[	-0.472906	,	0.842519	],
	[	0.061407	,	0.34504	],
	[	0.592726	,	-0.245692	],
	[	-0.996084	,	0.1717	],
	[	0.890094	,	0.430487	],
	[	-0.601101	,	0.808974	],
	[	0.238966	,	-0.044555	],
	[	1.64346	,	-0.317541	],
	[	0.732115	,	0.340259	],
	[	0.729599	,	-0.576685	],
	[	0.281293	,	-0.302384	],
	[	1.762774	,	-0.083138	],
	[	0.771201	,	0.771141	],
	[	0.151335	,	-0.06745	],
	[	-0.484603	,	0.929286	],
	[	1.049296	,	0.674038	],
	[	1.764874	,	-0.136656	],
	[	-0.124933	,	0.466371	],
	[	-0.720892	,	0.349213	],
	[	0.104445	,	0.39924	],
	[	-0.016942	,	0.48595	],
	[	-0.711277	,	0.84908	],
	[	0.995014	,	0.181074	],
	[	0.126856	,	0.116745	],
	[	-0.015635	,	0.275151	],
	[	0.612307	,	-0.340938	],
	[	0.678361	,	0.707514	],
	[	1.255764	,	-0.321081	],
	[	0.738453	,	-0.37601	]],
    np.double
	)