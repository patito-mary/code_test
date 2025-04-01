def dictionary_select_items(x, ind):
    '''This function takes a dictionary x and a boolean array ind.
    It returns a new dictionary with the same keys as x, but values
    are filtered by ind. It also adds a new key "indices" containing
    the indices where ind is True.'''
    z = {}
    for key in x:
        if key != 'count':
            z[key] = x[key][ind]
    z['count'] = np.sum(ind)
    z['halo_id'] = np.where(ind)[0]
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

def select_particles_within_radius_of_halos(Halos_cat, Halos_limits,
                                            r_multiplier,
                                            basePath, snapNum, partType):
 """
    Selects halos that meet specified criteria and loads their particles within a given radius.

    Args:
        Halos_cat (dict): Halo catalog.
        Halos_limits (dict): Dictionary specifying halo property limits.
        r_multiplier (float, optional): Multiple of R200 to consider. Defaults to 3.0.
        basePath (str): Base path to the simulation data.
        snapNum (int): Snapshot number.
        partType (int or str): Particle type.


    Returns:
        tuple: A tuple containing:
            - selected_halos (dict): Dictionary of selected halos.
            - particles_per_halo (list): List of dictionaries, each containing particle data for a selected halo.
    """   
    # 1. Seleccionar halos que cumplen los criterios
    halo_mask = select_halos(Halos_cat, Halos_limits)
    selected_halos = dictionary_select_items(Halos_cat, halo_mask)
    
    # 2. Para cada halo seleccionado, cargar y seleccionar sus partículas
    particles_per_halo = []
    
    for i in range(selected_halos['count']):
        halo_id = selected_halos['halo_id'][i]  
        halo_pos = selected_halos['GroupPos'][i]
        halo_r200 = selected_halos['Group_R_Crit200'][i]
        
        # Cargar partículas de este halo
        
        particle_data = il.snapshot.loadHalo(basePath, snapNum, id=halo_id, partType, fields=['Coordinates', 'Masses', 'ParticleIDs'])
        
        # Calcular distancias al centro del halo
        distances = np.linalg.norm(particle_data['Coordinates'] - halo_pos, axis=1)
        r_cut = r_multiplier * halo_r200
        
        # Crear máscara para partículas dentro del radio
        particle_mask = distances <= r_cut
        
        # Filtrar partículas y guardar
        selected_particles = dictionary_select_items(particle_data, particle_mask)
        selected_particles['halo_id'] = halo_id  # Añadir referencia al halo padre
        particles_per_halo.append(selected_particles)
    
    return selected_halos, particles_per_halo


def add_center_mass(aux_Halo_cat, basePath, Header, Halos_cat):
    i=0
    list_SZ = []
    #print(f'Computing the center of mass of the gas in the snapshot {N}')
    for NHalo in aux_Halo_cat['halo_id']:
        NHalo= int(NHalo)
        #print(f'Computing the center of mass of gas from the halo number:   {NHalo}')
        SZ=CM_3D(aux_Halo_cat['Coordinates'][i], aux_Halo_cat['Masses'][i], Halos_cat['GroupPos'][NHalo], Header['BoxSize'])
        list_SZ.append(SZ)   
        i+=1
    #print("This Snapshot have been finished")
    aux_Halo_cat = aux_Halo_cat | {'GroupSZPos': list_SZ}
    return aux_Halo_cat

def add_halo_center_masses(aux_Halo_cat, Halos_cat, box_size):
    center_of_mass_list = []
    for i, halo_id in enumerate(aux_Halo_cat['halo_id']):
        halo_id = int(halo_id)
        center_of_mass = CM_3D(aux_Halo_cat['Coordinates'][i], aux_Halo_cat['Masses'][i], Halos_cat['GroupPos'][halo_id], box_size)
        center_of_mass_list.append(center_of_mass)

    updated_halo_cat = aux_Halo_cat.copy()  # Create a copy to avoid modifying the original
    updated_halo_cat['GroupSZPos'] = center_of_mass_list
    return updated_halo_cat

def Distance_3D_vectors(vector1, vector2, BoxSize):
    ''' This function takes as input two 3D vectors of the same size and computes the 3D distance between those two vectors component by component. The output of this function is a 3D array containing the distance between those points '''
    if np.size(vector1)==np.size(vector2):
        for i in range(int(np.size(vector1)/3)): 
            try: aux_offset= np.append(aux_offset, [Distance_3D(vector1[i], vector2[i], BoxSize)], axis=0)
            except: aux_offset=Distance_3D(vector1[i], vector2[i], BoxSize).reshape(1,3)
    return aux_offset

def calculate_halo_offsets(halos_data, box_size):
    """Calculates the offset between GroupPos and GroupSZPos for each halo.

    Args:
        halos_data (DataFrame or dict): Halo catalog with 'GroupPos' and 'GroupSZPos' columns.
        box_size (float): Size of the simulation box.

    Returns:
        DataFrame or dict: Halo catalog with an additional 'Offset' column.
    """
    updated_halos_data = halos_data.copy()

    if isinstance(halos_data, pd.DataFrame):  # Handle pandas DataFrame
        updated_halos_data['Offset'] = updated_halos_data.apply(
            lambda row: Distance_3D_vectors(row['GroupPos'], row['GroupSZPos'], box_size), axis=1
        )
    elif isinstance(halos_data, dict):  # Handle dictionary-like data
        offsets = []
        for i in range(halos_data['count']):  # Assuming 'count' key exists for dictionaries
            offset = Distance_3D_vectors(halos_data['GroupPos'][i], halos_data['GroupSZPos'][i], box_size)
            offsets.append(offset)
        updated_halos_data['Offset'] = np.array(offsets)
    else:
        raise TypeError("halos_data must be a pandas DataFrame or a dictionary-like object.")

    return updated_halos_data