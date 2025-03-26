def dictionary_select_items(x, ind):
    '''This function needs as input a dictionary x and an array of indices ind
    and returns a new dictionary with all fields, but just with the selected data'''
    keys_to_update=[i for i in x.keys() -{'count'}]
    z=x
    for i in keys_to_update:
        z[i]=x[i][ind]
    z['count']= np.size(np.where(ind))
    return z

def select_halos_within_3r200(Halos_cat, Halos_limits, Header, n_R200, particle, N, basePath):
    """
    Selecciona particulas de los halos que cumplen con los límites de masa y están dentro de 3 R200
    respecto al centro de masas del halo.
    
    Parámetros:
    - Halos_cat: Catálogo de halos.
    - n_R200: En R200, distancia al CM del grupo de cada particula
    
    Retorna:
    - selected_halos: Un array booleano que indica qué halos cumplen las condiciones.
    """
    
    # Selección inicial basada en la masa (Group_M_Crit200)
    selected_halos = np.logical_and(
        Halos_cat['Group_M_Crit200'] >= Halos_limits["Group_M_Crit200"][0],
        Halos_cat['Group_M_Crit200'] <= Halos_limits["Group_M_Crit200"][1]
    )

    for_nR200_coords = []
    all_fof_coords = []
    halo_id = []
    
    gas_particles = {}
    
    for halo in np.where(selected_halos)[0]:
        print(f'loading data for halo: {halo} . . .')
        # Obtener la posición del halo principal (asumimos que es el primer halo en el catálogo)
        halo_cm_pos = Halos_cat['GroupCM'][halo]/Header['HubbleParam']
        halo_r200 = Halos_cat['Group_R_Crit200'][halo]/Header['HubbleParam']
        particle_gas = il.snapshot.loadHalo(basePath, N, id=halo, partType=particle, fields=['Masses','Coordinates'])
        gas_pos = particle_gas['Coordinates']/Header['HubbleParam']
        # Calcular la distancia de todas las particulas del fof al CM halo
        dist_to_principal = Distance_3D_to_1D(gas_pos, halo_cm_pos, Header['BoxSize'])
        norm_dist = dist_to_principal/halo_r200
        
        mask_NR200 = np.where(norm_dist <= n_R200)[0]
        use_gas = gas_pos[mask_NR200]

        if halo not in gas_particles:
            gas_particles[halo] = {}

        gas_particles[halo]['inside_3R200'] = use_gas
        gas_particles[halo]['all_fof'] = dictionary_select_items(particle_gas, mask_NR200)

        # Release memory for this halo
        del halo_cm_pos, halo_r200, particle_gas, gas_pos, dist_to_principal, norm_dist, mask_NR200, use_gas

    return gas_particles


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