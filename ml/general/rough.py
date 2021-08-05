
	# --- Plot --
	plt.setp(ax, xticks=(), yticks=());
	X_test = np.ndarray((n_samples,2), dtype = float);	# Matrix to store values (cols: x-values, y-values, identification, classification, verification).
	X_test = np.linspace(-1, 1, n_samples);			# Points for plotting.
	mesh = np.meshgrid(X_test,X_test);			# Mesh for 3D plotting.
	Z_true =  np.ndarray((len(X_test),len(X_test)), dtype = float);	
	Z_model = np.ndarray((len(X_test),len(X_test)), dtype = float);
	tmp = np.ndarray((len(X_test),2), dtype = float);
	for i in range(0,len(X_test)):
		tmp[:,0] = mesh[1][i];
		tmp[:,1] = mesh[0][i];
		Z_true[:,i] = linear_regression_true.predict(tmp);       
		Z_model[:,i] = linear_regression_model.predict(tmp);       

	pts = pd.DataFrame(pts);				# Convert to DataFrame.

	ax.plot_surface(mesh[0], mesh[1], Z_true, rstride = 100, cstride = 100,color = 'blue');     # Plot True plane.
#ax.plot_surface(mesh[0], mesh[1], Z_model, rstride = 10, cstride = 10, color = 'black');  # Plot Model plane.
	ax.view_init(elev=90);
	plt.xlim((-1, 1));	# Limits: x-axis [-1,1].
	plt.ylim((-1, 1));	# Limits: y-axis [-1,1].
	plt.xlabel("x");	# Lable: x-axis.
	plt.ylabel("y");	# Lable: y-axis.
	plt.legend(loc="best"); # Legend.
	plt.show();		# Show plot.
	# -----
