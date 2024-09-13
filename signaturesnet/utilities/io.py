import logging
import os
import pathlib
from sklearn import preprocessing
import sys
import yaml

import json
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signaturesnet import DATA, TRAINED_MODELS
from signaturesnet.utilities.data_partitions import DataPartitions
from signaturesnet.utilities.generator_data import GeneratorData

def read_signatures(file,
                    mutation_type_order=os.path.join(DATA, "mutation_type_order.xlsx")):
    """
    File must contain first column with mutations types X[Y>Z]W and the rest of the columns must be the set of signatures
    """
    # Sort according to cosmic mutation types order
    signatures_data = sort_signatures(file,
                                      mutation_type_order=mutation_type_order)

    num_sigs = len(signatures_data.columns) - 1
    signatures = [torch.tensor(signatures_data.iloc[:, i]).type(torch.float32)
                  for i in range(1, num_sigs + 1)][:num_sigs]
    signatures = torch.stack(signatures).t()
    return signatures

def sort_signatures(file,
                    output_file=None,
                    mutation_type_order=os.path.join(DATA, "mutation_type_order.xlsx")):
    signatures_data = pd.read_excel(file)
    mutation_order = pd.read_excel(mutation_type_order)

    # Sort according to cosmic mutation types order
    signatures_data.rename(columns = {list(signatures_data)[0]:'Type'}, inplace=True)
    signatures_data = signatures_data.set_index('Type')
    signatures_data = signatures_data.reindex(index=mutation_order['Type'])
    signatures_data = signatures_data.reset_index()

    if output_file is not None:
        create_dir(output_file)
        signatures_data.to_csv(output_file, index=False)
    return signatures_data

def csv_to_pandas(file,
                  device="cpu",
                  header=None,
                  index_col=None,
                  type_df=None):
    df = pd.read_csv(file, header=header, index_col=index_col)
    df.index = df.index.map(lambda x: x.split("..")[-1])

    if type_df is not None:
        cancer_type_df = pd.read_csv(type_df, header=0)[["Cancer Types", "Sample Names"]]
        df = df.merge(cancer_type_df, left_index=True, right_on="Sample Names").set_index("Sample Names")
        le = preprocessing.LabelEncoder()
        le.fit(df["Cancer Types"])
        df['cancer_type'] = le.transform(df["Cancer Types"])
        df = df.drop("Cancer Types", axis=1)
    return df

def csv_to_tensor(file,
                  device="cpu",
                  header=None,
                  index_col=None,
                  type_df=None,):
    df = pd.read_csv(file, header=header, index_col=index_col)

    if type_df is not None:
        df.index = df.index.map(lambda x: x.split("..")[-1])
        cancer_type_df = pd.read_csv(type_df, header=0)[["Cancer Types", "Sample Names"]]
        df = df.merge(cancer_type_df, left_index=True, right_on="Sample Names").set_index("Sample Names")
        le = preprocessing.LabelEncoder()
        le.fit(df["Cancer Types"])
        df['cancer_type'] = le.transform(df["Cancer Types"])
        df = df.drop("Cancer Types", axis=1)

    input_tensor = torch.tensor(df.values, dtype=torch.float)
    assert(not torch.isnan(input_tensor).any())
    return input_tensor.float().to(device)

def tensor_to_csv(data_tensor, output_path):
    create_dir(output_path)
    df = data_tensor.detach().cpu().numpy()
    df = pd.DataFrame(df)
    df.to_csv(output_path, header=False, index=False) 

def read_data(device,
              experiment_id,
              source,
              data_folder=DATA,
              include_baseline=True,
              include_labels=True,
              n_points = None):
    """Read data from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        source (string): Type of generated data: random or realistic
        data_folder (str, optional): Relative path of data folder. Defaults to DATA.
    """
    # assert(source in ["random", "realistic", "perturbed"])
    path = os.path.join(data_folder, experiment_id)

    train_input = csv_to_tensor(path + "/train_%s_input.csv" % source, device)
    train_baseline = csv_to_tensor(path + "/train_%s_baseline.csv" % source, device) if include_baseline else None
    train_label = csv_to_tensor(path + "/train_%s_label.csv" % source, device) if include_labels else None
    train_data = DataPartitions(inputs=train_input,
                                prev_guess=train_baseline,
                                labels=train_label)
    train_data.perm
    if n_points is not None:
        train_data.inputs = train_data.inputs[:n_points,:]
        train_data.prev_guess = train_data.prev_guess[:n_points,:]
        train_data.labels = train_data.labels[:n_points,:]

    val_input = csv_to_tensor(path + "/val_%s_input.csv" % source, device)
    val_baseline = csv_to_tensor(path + "/val_%s_baseline.csv" % source, device) if include_baseline else None
    val_label = csv_to_tensor(path + "/val_%s_label.csv" % source, device) if include_labels else None
    
    val_data = DataPartitions(inputs=val_input,
                              prev_guess=val_baseline,
                              labels=val_label)

    return train_data, val_data

def read_data_classifier(device,
                         experiment_id,
                         data_folder=DATA):
    """Read data from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        source (string): Type of generated data: random or realistic
        data_folder (str, optional): Relative path of data folder. Defaults to "DATA".
    """
    path = os.path.join(data_folder, experiment_id)

    train_input = csv_to_tensor(path + "/train_input.csv", device)
    train_num_mut = csv_to_tensor(path + "/train_num_mut.csv", device).reshape((-1,1))
    train_label = csv_to_tensor(path + "/train_label.csv", device).reshape((-1,1))

    train_data = DataPartitions(inputs=train_input,
                                num_mut=train_num_mut,
                                labels=train_label)

    val_input = csv_to_tensor(path + "/val_input.csv", device)
    val_num_mut = csv_to_tensor(path + "/val_num_mut.csv", device).reshape((-1,1))
    val_label = csv_to_tensor(path + "/val_label.csv", device).reshape((-1,1))

    val_data = DataPartitions(inputs=val_input,
                              num_mut=val_num_mut,
                              labels=val_label)

    return train_data, val_data

def read_real_data(device, experiment_id, data_folder=DATA):
    """Read data from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        source (string): Type of generated data: random or realistic
        data_folder (str, optional): Relative path of data folder. Defaults to DATA.
    """
    path = os.path.join(data_folder, experiment_id)

    real_input = csv_to_tensor(path + "/real_data_input.csv", device)
    real_num_mut = csv_to_tensor(path + "/real_data_num_mut.csv", device)

    return real_input, real_num_mut


def read_data_generator(device,
                        data_id,
                        data_folder=DATA,
                        cosmic_version='v3',
                        type='real',
                        prop_train=0.9):
    '''
    type should be: 'real', 'perturbed' or 'augmented_real'.
    '''
    data_folder = data_folder + data_id
    if type == 'real':
        if cosmic_version == 'v3':
            real_data = csv_to_pandas(data_folder + "/sigprofiler_not_norm_PCAWG.csv",
                                    device=device,
                                    header=0,
                                    index_col=0,
                                    type_df=data_folder + "/PCAWG_sigProfiler_SBS_signatures_in_samples_v3.csv")
            
            num_ctypes = real_data['cancer_type'][-1]+1
            real_data = real_data.groupby('cancer_type').sample(frac=1, random_state=0)       #Shuffle samples inside the same cancer type
            # print(real_data.size(0)/num_ctypes*prop_train)
            real_data_train = real_data.groupby('cancer_type').head(int(round(real_data.shape[0]/num_ctypes*prop_train)))  #Take the first prop_train % of samples in each cancer type
            real_data_rest = pd.concat([real_data, real_data_train]).drop_duplicates(keep=False)                    # The rest is for validation and testing

            real_data_train = torch.tensor(real_data_train.values, dtype=torch.float)
            real_data_rest = torch.tensor(real_data_rest.values, dtype=torch.float)

            train_input, train_cancer_types = real_data_train[:, :-1], real_data_train[:, -1]
            train_input = train_input/torch.sum(train_input, axis=1).reshape(-1, 1)
            train_input = torch.cat([train_input, torch.zeros(train_input.size(0), 7).to(train_input)], dim=1)
            train_data = GeneratorData(inputs=train_input, cancer_types=train_cancer_types)

            val_input, val_cancer_types = real_data_rest[:, :-1], real_data_rest[:, -1]
            val_input = val_input/torch.sum(val_input, axis=1).reshape(-1, 1)
            val_input = torch.cat([val_input, torch.zeros(val_input.size(0), 7).to(val_input)], dim=1)
            val_data = GeneratorData(inputs=val_input, cancer_types=val_cancer_types)

        elif cosmic_version == 'v2':
            real_data = csv_to_tensor(data_folder + "/PCAWG_genome_deconstructSigs_v2.csv",
                                    device=device, header=0, index_col=0)
            real_data = real_data/torch.sum(real_data, axis=1).reshape(-1, 1)

            perm = torch.randperm(real_data.size(0))
            data = real_data[perm, :]

            train_input = data[:int(real_data.size(0)*0.95)]
            val_input = data[int(real_data.size(0)*0.95):]

            train_data = GeneratorData(inputs=train_input)
            val_data = GeneratorData(inputs=val_input)

        else:
            raise NotImplementedError

        
    else:
        train_input = csv_to_tensor(data_folder + "/train_%s_low_label.csv"%type,
                                    device=device, header=None, index_col=None)
        val_input = csv_to_tensor(data_folder + "/val_%s_low_label.csv"%type,
                                    device=device, header=None, index_col=None)
        train_data = GeneratorData(inputs=train_input[:,:-1])
        val_data = GeneratorData(inputs=val_input[:,:-1])

    train_data.to(device)
    val_data.to(device)
    return train_data, val_data

def read_methods_guesses(device, experiment_id, methods, data_folder=DATA):
    """Read one method guess from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        test_id (string): Full name of the test folder
        method (list): List of string with the methods to be analyzed
        data_folder (str, optional): Relative path of data folder. Defaults to DATA.
    """
    if experiment_id is not None:
        path = os.path.join(data_folder, experiment_id)

        methods_guesses = []
        for method in methods:
            methods_guesses.append(csv_to_tensor(
                path + "/other_methods/all_results/%s_guess.csv" % (method), device))

        label = csv_to_tensor(path + "/test_label.csv", device)

        return methods_guesses, label
    else:
        methods_guesses = []
        for method in methods:
            methods_guesses.append(csv_to_tensor(
                data_folder + "/%s_guess.csv" % (method), device))

        label = csv_to_tensor(DATA + "/datasets/test_label.csv", device)

        return methods_guesses, label


def read_test_data(device, experiment_id, test_id, data_folder=DATA):
    """Read one method guess from disk

    Args:
        device (string): Device to train on
        experiment_id (string): Full name of the experiment folder
        test_id (string): Full name of the test folder
        data_folder (str, optional): Relative path of data folder. Defaults to DATA.
    """
    path = os.path.join(data_folder, experiment_id, test_id)

    inputs = csv_to_tensor(path + "/%s_input.csv" % (test_id), device=device)
    label = csv_to_tensor(path + "/%s_label.csv" % (test_id), device=device)

    return inputs, label


def read_model(directory, device="cpu"):
    """Instantiate a pre-trained model from the stored vars
    The model is in cpu and in eval mode

    Args:
        directory (String): Folder containing state_dict and init_args.json of the model
    """
    from signaturesnet.models import Generator, Classifier, FineTunerLowNumMut, FineTunerLargeNumMut, ErrorFinder, NumMutNet, VaeClassifier

    # Load init_args
    init_args_file = os.path.join(directory, 'init_args.json')
    with open(init_args_file, 'r') as fp:
        init_args = json.load(fp)
    model_type = init_args["model_type"]
    init_args.pop("model_type")
    # print("Reading model of type:", model_type)
    assert(model_type is not None)  # Model type not saved!
    assert(model_type in ["Classifier", "FineTunerLowNumMut", "FineTunerLargeNumMut", "ErrorFinder", "Generator", "NumMutNet", "VaeClassifier"])
    if "device" in init_args.keys():
        init_args["device"] = device
        
    # Instantiate model class
    if model_type == "Generator":
        model = Generator(**init_args)
    if model_type == "Classifier":
        model = Classifier(**init_args)
    elif model_type == "FineTunerLowNumMut":
        model = FineTunerLowNumMut(**init_args)
    elif model_type == "FineTunerLargeNumMut":
        model = FineTunerLargeNumMut(**init_args)
    elif model_type == "ErrorFinder":
        model = ErrorFinder(**init_args)
    elif model_type == "NumMutNet":
        model = NumMutNet(**init_args)
    elif model_type == "VaeClassifier":
        model = VaeClassifier(**init_args)
    
    # Load model weights
    state_dict_file = os.path.join(directory, "state_dict")
    try:
        state_dict = torch.load(f=state_dict_file,
                                map_location=torch.device(device))
    except:
        state_dict = torch.load(f=state_dict_file + ".zip",
                                map_location=torch.device(device))

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def save_model(model, directory):
    """Store a pytorch model. The arguments are splitted into 2 files:
    A init_args.json needed to instantiate the class, and the model state_dict

    Args:
        model (nn.Module): Model to save
        directory (String): Path where to save
    """
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Store init_args into a json file
    init_args_file = os.path.join(directory, 'init_args.json')
    with open(init_args_file, 'w') as fp:
        json.dump(model.init_args, fp)

    # Store state_dict
    state_dict_file = os.path.join(directory, "state_dict")
    torch.save(model.state_dict(), state_dict_file)


def update_dict(config, args):
    for arg in vars(args):
        if getattr(args, arg) is not None and arg in config:
            if len(getattr(args, arg)) == 1:
                config[arg] = getattr(args, arg)[0]
            else:
                config[arg] = getattr(args, arg)
    return config

def read_config(path):
    with open(path, 'r') as stream:
        data = yaml.safe_load(stream)
    return data["config"]

def create_dir(filepath):
    directory = os.path.dirname(filepath)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

def write_result(result, filepath):
    directory = os.path.dirname(filepath)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    fout = open(filepath, 'w')
    fout.write(str(result))
    fout.close()

def write_final_outputs(weights,
                        lower_bound,
                        upper_bound,
                        classification,
                        sample_names, 
                        output_path,
                        name=''):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    sig_names = list(pd.read_excel(os.path.join(DATA, "data.xlsx")).columns)[1:]

    def write(values, path, sig_names, output_type = 'error'):
        logging.info("Writting results: %s"%path)
        try:
            df = pd.DataFrame(values.detach().numpy())
        except:
            df = pd.DataFrame(values)
        if output_type == 'weight':
            sig_names = sig_names + ['Unknown']
        if output_type == 'classification':
            sig_names = ['Classification']
        df.columns = sig_names
        df.index = sample_names
        df.to_csv(path, header=True, index=True)

    # Write results weight guesses
    write(values=weights, path=output_path + "/weight_guesses-%s.csv"%name, sig_names=sig_names, output_type='weight')
    write(values=lower_bound, path=output_path + "/lower_bound_guesses-%s.csv"%name, sig_names=sig_names)
    write(values=upper_bound, path=output_path + "/upper_bound_guesses-%s.csv"%name, sig_names=sig_names)
    write(values=classification, path=output_path + "/classification_guesses-%s.csv"%name, sig_names=sig_names, output_type='classification')


def write_David_outputs(weights, lower_bound, upper_bound, output_path):
    sig_names = list(pd.read_excel(os.path.join(DATA, "data.xlsx")).columns)[1:]
    
    # Write results weight guesses
    df = pd.DataFrame({'weight_guess': weights[0], 'upper_bound': upper_bound[0], 'lower_bound': lower_bound[0],})
    df.index = sig_names
    df.to_csv(output_path + "_guess.csv", header=True, index=True)

