def dictionary_select_items(x, ind):
    '''This function needs as input a dictionary x and an array of indices ind
    and returns a new dictionary with all fields, but just with the selected data'''
    keys_to_update=[i for i in x.keys() -{'count'}]
    z=x
    for i in keys_to_update:
        z[i]=x[i][ind]
    z['count']= np.size(np.where(ind))
    return z
def select_halos(Halos_cat, Halos_limits):
    """
    Selecciona halos que cumplen con los límites especificados.
    
    Parámetros:
    - Halos_cat: Catálogo de halos.
    - Halos_limits: Diccionario con límites para las propiedades de los halos.
    
    Retorna:
    - selected_halos: Array booleano que indica qué halos cumplen con los límites.
    """
    selected_halos = np.ones(len(Halos_cat['Group_R_Crit200']), dtype=bool)  # Inicializar con todos los halos seleccionados
    
    for i in Halos_limits.keys():
        aux_key = i.split('/')
        if np.size(aux_key) == 1:
            selected_halos = np.logical_and(
                selected_halos,
                np.logical_and(
                    Halos_cat[i] >= Halos_limits[i][0],
                    Halos_cat[i] <= Halos_limits[i][1]
                )
            )
        else:
            selected_halos = np.logical_and(
                selected_halos,
                np.logical_and(
                    Halos_cat[aux_key[0]][:, int(aux_key[1])] >= Halos_limits[i][0],
                    Halos_cat[aux_key[0]][:, int(aux_key[1])] <= Halos_limits[i][1]
                )
            )
    
    return selected_halos

def select_particles_in_halos(Halos_cat, Particles_cat, Halos_limits, r_multiplier=3.0):
    """
    Selecciona halos que cumplen con los límites y luego sus partículas dentro de r_multiplier×R200
    
    Parámetros:
    - Halos_cat: Catálogo de halos
    - Particles_cat: Catálogo de partículas
    - Halos_limits: Diccionario con límites para propiedades de halos
    - r_multiplier: Múltiplo del radio R200 a considerar (default 3.0)
    
    Retorna:
    - selected_halos: Halos que cumplen los criterios
    - selected_particles: Partículas dentro del radio especificado
    """
    # 1. Seleccionar halos que cumplen los criterios
    halo_mask = select_halos(Halos_cat, Halos_limits)
    selected_halos = dictionary_select_items(Halos_cat, halo_mask)
    
    # 2. Para cada halo seleccionado, encontrar partículas dentro de 3×R200
    particle_mask = np.zeros(len(Particles_cat['Coordinates']), dtype=bool)
    
    for i in range(len(selected_halos['Group_R_Crit200'])):
        # Calcular distancia de cada partícula al centro del halo
        halo_pos = selected_halos['GroupPos'][i]
        particle_pos = Particles_cat['Coordinates']
        distances = np.sqrt(np.sum((particle_pos - halo_pos)**2, axis=1))
        
        # Radio de corte para este halo
        r_cut = r_multiplier * selected_halos['Group_R_Crit200'][i]
        
        # Actualizar máscara de partículas
        particle_mask = np.logical_or(particle_mask, distances <= r_cut)
    
    # 3. Filtrar las partículas seleccionadas
    selected_particles = dictionary_select_items(Particles_cat, particle_mask)
    
    return selected_halos, selected_particles