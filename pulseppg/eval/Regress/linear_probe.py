from pulseppg.eval.Base_Eval import Base_EvalClass
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import joblib
from pulseppg.utils.utils import printlog

from sklearn import metrics
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error, mean_poisson_deviance


class Model(Base_EvalClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_eval(self, **kwargs):
        super().setup_eval(**kwargs)
        for param in self.trained_net.parameters():
            param.requires_grad = False

    def fit(self):
        printlog(f"Begin Training {self.model_file}", self.run_dir)

        writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb"))

        num_train = self.train_data.shape[0]
        num_val = self.val_data.shape[0]

        # Save raw training and validation data
        try:
            train_data_np = self.train_data.cpu().numpy() if isinstance(self.train_data, torch.Tensor) else self.train_data
            val_data_np = self.val_data.cpu().numpy() if isinstance(self.val_data, torch.Tensor) else self.val_data
            
            np.save(os.path.join(self.run_dir, "raw_train_data.npy"), train_data_np)
            np.save(os.path.join(self.run_dir, "raw_train_labels.npy"), self.train_labels)
            np.save(os.path.join(self.run_dir, "raw_val_data.npy"), val_data_np)
            np.save(os.path.join(self.run_dir, "raw_val_labels.npy"), self.val_labels)
            
            printlog(f"Saved raw_train_data.npy: {train_data_np.shape}", self.run_dir)
            printlog(f"Saved raw_train_labels.npy: {self.train_labels.shape}", self.run_dir)
            printlog(f"Saved raw_val_data.npy: {val_data_np.shape}", self.run_dir)
            printlog(f"Saved raw_val_labels.npy: {self.val_labels.shape}", self.run_dir)
        except Exception as e:
            printlog(f"Warning: could not save raw data: {e}", self.run_dir)

        # Generate and save separate train embeddings
        X_train = self._generate_embeddings(self.train_data, batch_size=128)
        try:
            np.save(os.path.join(self.run_dir, "embeddings_train.npy"), X_train)
            printlog(f"Saved embeddings_train.npy: {X_train.shape}", self.run_dir)
        except Exception as e:
            printlog(f"Warning: could not save train embeddings: {e}", self.run_dir)

        # Generate and save separate val embeddings
        X_val = self._generate_embeddings(self.val_data, batch_size=128)
        try:
            np.save(os.path.join(self.run_dir, "embeddings_val.npy"), X_val)
            printlog(f"Saved embeddings_val.npy: {X_val.shape}", self.run_dir)
        except Exception as e:
            printlog(f"Warning: could not save val embeddings: {e}", self.run_dir)

        # Concatenate for trainval
        X_trainval = np.concatenate([X_train, X_val])
        y_trainval = np.concatenate((self.train_labels, self.val_labels))

        # Save combined trainval embeddings and labels for analysis
        try:
            np.save(os.path.join(self.run_dir, "embeddings_trainval.npy"), X_trainval)
            np.save(os.path.join(self.run_dir, "labels_trainval.npy"), y_trainval)
            printlog(f"Saved embeddings_trainval.npy: {X_trainval.shape}", self.run_dir)
            printlog(f"Saved labels_trainval.npy: {y_trainval.shape}", self.run_dir)
        except Exception as e:
            printlog(f"Warning: could not save trainval data: {e}", self.run_dir)

        scaler = StandardScaler()
        X_trainval = scaler.fit_transform(X_trainval)

        estimator = Ridge()
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0],  # Regularization strength
            'solver': ['auto', 'cholesky', 'sparse_cg']  # Solver to use in the computational routines
        }

        grid_search = GridSearchCV(estimator=estimator, 
                            param_grid=param_grid, 
                            cv=4, 
                            scoring='neg_mean_squared_error', 
                            verbose=self.config.verbose, 
                            n_jobs=self.config.num_threads)
        grid_search.fit(X_trainval, y_trainval)

        printlog(f"Finished Training {self.model_file}", self.run_dir)

        joblib.dump(grid_search, f"{self.run_dir}/checkpoint_cv_best.joblib")
        joblib.dump(scaler, f"{self.run_dir}/checkpoint_scaler_best.joblib")
        state_dict = {"trained_net": self.trained_net.state_dict()}
        torch.save(state_dict, f"{self.run_dir}/checkpoint_best.pkl")

    def _generate_embeddings(self, data: torch.Tensor, batch_size: int = 128) -> np.ndarray:
        """Generate embeddings for input data using the trained network."""
        embeddings = []
        self.trained_net.eval()
        with torch.no_grad():
            for i in tqdm(range(0, data.shape[0], batch_size)):
                batch = data[i : i + batch_size].cuda()
                embeddings.append(
                    self.trained_net(batch)
                    .cpu()
                    .detach()
                    .numpy()
                )
        return np.concatenate(embeddings)

    def load(self):
        state_dict = torch.load(
            f"{self.run_dir}/checkpoint_best.pkl", map_location=self.device
        )

        print(self.trained_net.load_state_dict(state_dict["trained_net"]))
        self.grid_search = joblib.load(f"{self.run_dir}/checkpoint_cv_best.joblib")
        self.scaler = joblib.load(f"{self.run_dir}/checkpoint_scaler_best.joblib")

        printlog(f"Reloading {self.model_file} Model's CV", self.run_dir)

    def test(self, test_idx=None, dontprint=False):
        printlog(f"Loading Best From Training", self.run_dir)
        self.load()

        writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb"))

        X_test = torch.Tensor(self.test_data)
        y_test = self.test_labels

        # Save raw test data
        try:
            test_data_np = self.test_data.cpu().numpy() if isinstance(self.test_data, torch.Tensor) else self.test_data
            
            np.save(os.path.join(self.run_dir, "raw_test_data.npy"), test_data_np)
            np.save(os.path.join(self.run_dir, "raw_test_labels.npy"), self.test_labels)
            printlog(f"Saved raw_test_data.npy: {test_data_np.shape}", self.run_dir)
            printlog(f"Saved raw_test_labels.npy: {self.test_labels.shape}", self.run_dir)
        except Exception as e:
            printlog(f"Warning: could not save raw test data: {e}", self.run_dir)

        # Generate and save test embeddings
        X_test_embeddings = self._generate_embeddings(X_test, batch_size=128)
        try:
            np.save(os.path.join(self.run_dir, "embeddings_test.npy"), X_test_embeddings)
            printlog(f"Saved embeddings_test.npy: {X_test_embeddings.shape}", self.run_dir)
        except Exception as e:
            printlog(f"Warning: could not save test embeddings: {e}", self.run_dir)

        # Save test labels for analysis
        try:
            np.save(os.path.join(self.run_dir, "labels_test.npy"), y_test)
            printlog(f"Saved labels_test.npy: {y_test.shape}", self.run_dir)
        except Exception as e:
            printlog(f"Warning: could not save test labels: {e}", self.run_dir)

        X_test_scaled = self.scaler.transform(X_test_embeddings)
        y_pred = self.grid_search.predict(X_test_scaled)

        if test_idx is not None:
            print(f"Predicted value is {y_pred[test_idx]}")

        unique_labels = []
        mean_predictions = []

        for label in np.unique(y_test):
            mask = (y_test == label)
            mean_pred = y_pred[mask].mean()

            unique_labels.append(label)
            mean_predictions.append(mean_pred)

        # -------------------------
        # 3. Compute MAE
        # -------------------------
        mae = mean_absolute_error(unique_labels, mean_predictions)
        print("MAE unique labels:", mae)


        # Calculate the metrics
        total_mae = mean_absolute_error(y_test, y_pred)
        total_sdae = standard_deviation_of_absolute_error(y_test, y_pred)
        total_mse = mean_squared_error(y_test, y_pred)
        total_sdse = standard_deviation_of_squared_error(y_test, y_pred)
        # Mean Error (signed) and its standard deviation
        # ME is the average residual (prediction - truth), showing bias
        total_me = np.mean(np.array(y_pred) - np.array(y_test))
        total_sde = np.std(np.array(y_pred) - np.array(y_test))
        total_r2 = r2_score(y_test, y_pred)
        total_mape = mean_absolute_percentage_error(y_test, y_pred)

        # Build the printout string
        printoutstring = f"MAE/Test={total_mae:5f}\n"
        writer.add_scalar('MAE/Test', total_mae, 0)

        printoutstring += f"SDAE/Test={total_sdae:5f}\n"
        writer.add_scalar('SDAE/Test', total_sdae, 0)

        printoutstring += f"MSE/Test={total_mse:5f}\n"
        writer.add_scalar('MSE/Test', total_mse, 0)

        printoutstring += f"SDSE/Test={total_sdse:5f}\n"
        writer.add_scalar('SDSE/Test', total_sdse, 0)

        printoutstring += f"ME/Test={total_me:5f}\n"
        writer.add_scalar('ME/Test', total_me, 0)

        printoutstring += f"SDE/Test={total_sde:5f}\n"
        writer.add_scalar('SDE/Test', total_sde, 0)

        printoutstring += f"R2/Test={total_r2:5f}\n"
        writer.add_scalar('R2/Test', total_r2, 0)

        printoutstring += f"MAPE/Test={total_mape:5f}\n"
        writer.add_scalar('MAPE/Test', total_mape, 0)
        # Log the metrics
        printlog(printoutstring, self.run_dir, dontprint=dontprint)

        # Return metrics as a dictionary
        return {
            "MAE": total_mae,
            "SDAE": total_sdae, 
            "MSE": total_mse,
            "SDSE": total_sdse,  # Include SDSE
            "R2": total_r2,
            "MAPE": total_mape,
        }

def standard_deviation_of_absolute_error(true_values, predicted_values):
    # Calculate absolute errors
    absolute_errors = np.abs(np.array(true_values) - np.array(predicted_values))
    
    # Calculate and return the standard deviation of absolute errors
    return np.std(absolute_errors)

def standard_deviation_of_squared_error(true_values, predicted_values):
    # Calculate squared errors
    squared_errors = np.square(np.array(true_values) - np.array(predicted_values))
    
    # Calculate and return the standard deviation of squared errors
    return np.std(squared_errors)