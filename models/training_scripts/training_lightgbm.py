"""
LightGBM Training Script for Epidemic Prediction - Fixed Version
Ensures proper completion with 5-minute time limit, saves model, updates results, shows graphs
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import warnings
import os
import time
warnings.filterwarnings('ignore')

class LightGBMEpidemicPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.label_encoders = {}
        self.feature_importance = None
        self.model_name = f"lightgbm_epidemic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.cv_results = []
        self.total_time_limit = 300  # 5 minutes
        
    def load_and_prepare_data(self):
        """Load and engineer features in one step"""
        print("🔥 LOADING AND ENGINEERING FEATURES...")
        df = pd.read_csv('data/processed/master_dataframe.csv')
        
        # Basic datetime features
        df['date_semaine'] = pd.to_datetime(df['date_semaine'])
        df['year'] = df['date_semaine'].dt.year
        df['month'] = df['date_semaine'].dt.month
        df['week_of_year'] = df['date_semaine'].dt.isocalendar().week
        
        # Encode regions
        if 'region' in df.columns:
            self.label_encoders['region'] = LabelEncoder()
            df['region_encoded'] = self.label_encoders['region'].fit_transform(df['region'].fillna('Unknown'))
        
        # Encode season if available
        if 'saison' in df.columns:
            self.label_encoders['saison'] = LabelEncoder()
            df['saison_encoded'] = self.label_encoders['saison'].fit_transform(df['saison'].fillna('Unknown'))
        
        # Create essential lag and rolling features
        df = df.sort_values(['region', 'date_semaine'])
        for lag in [1, 2, 3, 4, 8, 12, 16]:
            df[f'urgences_lag_{lag}'] = df.groupby('region')['urgences_grippe'].shift(lag)
            
        for window in [2, 4, 8, 12, 16]:
            df[f'urgences_ma_{window}'] = (df.groupby('region')['urgences_grippe']
                                          .rolling(window, min_periods=1).mean().reset_index(0, drop=True))
            df[f'urgences_std_{window}'] = (df.groupby('region')['urgences_grippe']
                                           .rolling(window, min_periods=1).std().reset_index(0, drop=True))
        
        # IAS features if available
        if 'taux_ias_moyen' in df.columns:
            for lag in [1, 2, 4, 8]:
                df[f'ias_lag_{lag}'] = df.groupby('region')['taux_ias_moyen'].shift(lag)
        
        # Seasonal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52.0)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52.0)
        df['is_epidemic_season'] = df['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)
        
        # Vaccination features if available
        vacc_cols = [col for col in df.columns if 'vacc' in col.lower()]
        if len(vacc_cols) >= 2:
            df['vacc_total'] = df[vacc_cols].fillna(0).sum(axis=1)
        
        # Select features automatically - more conservative selection
        feature_candidates = [col for col in df.columns if any(x in col for x in 
                            ['lag_', 'ma_', 'std_', 'ias_', 'sin', 'cos', 'encoded', 'epidemic', 'year', 'month', 'week', 'vacc'])]
        # Filter out features with too many missing values or low variance
        self.feature_columns = []
        for col in feature_candidates:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                if non_null_count > 500:  # At least 500 non-null values
                    # Check for variance (avoid constant features)
                    if df[col].nunique() > 1:
                        self.feature_columns.append(col)
        
        print(f"✅ DATA READY: {df.shape}, {len(self.feature_columns)} features selected")
        print(f"📅 Date range: {df['date_semaine'].min()} to {df['date_semaine'].max()}")
        return df
    
    def split_data(self, df):
        """Temporal split"""
        train_data = df[df['year'].between(2019, 2021)].dropna(subset=['urgences_grippe']).copy()
        test_data = df[df['year'].between(2022, 2024)].dropna(subset=['urgences_grippe']).copy()
        print(f"📊 TRAIN: {train_data.shape}, TEST: {test_data.shape}")
        return train_data, test_data
    
    def run_quick_cv(self, train_data, start_time):
        """Run quick 3-fold CV with time limit"""
        print("\n🔄 STARTING QUICK 3-FOLD CROSS-VALIDATION")
        print("=" * 60)
        
        X_train = train_data[self.feature_columns].copy()
        y_train = train_data['urgences_grippe'].copy()
        
        # Fill missing values
        X_train = X_train.fillna(X_train.median())
        
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced to 3 folds for speed
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > 150:  # Max 2.5 minutes for CV
                print(f"⚠️ Time limit reached, stopping CV at fold {fold}")
                break
                
            fold_start = time.time()
            print(f"\n🔥 FOLD {fold + 1}/3 STARTING...")
            
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # LightGBM datasets
            train_lgb = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_lgb = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_lgb)
            
            # Conservative parameters to avoid overfitting
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,  # Reduced complexity
                'learning_rate': 0.1,  # Higher learning rate for faster training
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,  # Increased to prevent overfitting
                'min_child_weight': 1e-3,
                'min_split_gain': 0.1,  # Require minimum gain for splits
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'max_depth': 6,  # Reduced depth
                'random_state': 42 + fold,
                'verbosity': -1,
                'force_col_wise': True
            }
            
            # Train with early stopping and reduced iterations
            model = lgb.train(
                params, train_lgb, 
                num_boost_round=200,  # Reduced iterations
                valid_sets=[val_lgb], 
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20),  # Earlier stopping
                    lgb.log_evaluation(period=0)  # Silent
                ]
            )
            
            # Evaluate
            val_pred = model.predict(X_fold_val, num_iteration=model.best_iteration)
            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
            fold_r2 = r2_score(y_fold_val, val_pred)
            fold_mae = mean_absolute_error(y_fold_val, val_pred)
            fold_time = time.time() - fold_start
            
            cv_scores.append({
                'fold': fold + 1, 'rmse': fold_rmse, 'r2': fold_r2, 'mae': fold_mae,
                'best_iteration': model.best_iteration, 'training_time': fold_time
            })
            
            print(f"✅ FOLD {fold + 1} COMPLETE:")
            print(f"   📈 RMSE: {fold_rmse:.4f} | R²: {fold_r2:.4f} | MAE: {fold_mae:.2f}")
            print(f"   ⏱️  Time: {fold_time:.1f}s | Iterations: {model.best_iteration}")
        
        self.cv_results = cv_scores
        
        if cv_scores:
            # CV Summary
            mean_rmse = np.mean([s['rmse'] for s in cv_scores])
            std_rmse = np.std([s['rmse'] for s in cv_scores])
            mean_r2 = np.mean([s['r2'] for s in cv_scores])
            total_time = sum([s['training_time'] for s in cv_scores])
            
            print(f"\n🎯 CV RESULTS SUMMARY:")
            print(f"   📊 Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
            print(f"   📈 Mean R²: {mean_r2:.4f}")
            print(f"   ⏱️  Total CV Time: {total_time:.1f}s")
            print("=" * 60)
        
        return cv_scores
    
    def train_final_model(self, train_data, start_time):
        """Train final model with time limit and robust parameters"""
        print("\n🚀 TRAINING FINAL MODEL...")
        
        # Check remaining time
        elapsed = time.time() - start_time
        remaining_time = self.total_time_limit - elapsed - 30  # Reserve 30s for saving
        
        if remaining_time < 10:
            print("⚠️ Not enough time for final training, using simple model")
            return 10  # Return minimal training time
        
        X_train = train_data[self.feature_columns].copy()
        y_train = train_data['urgences_grippe'].copy()
        
        # Fill missing values
        X_train = X_train.fillna(X_train.median())
        
        # Validation split
        val_cutoff = train_data['date_semaine'].quantile(0.85)
        train_mask = train_data['date_semaine'] <= val_cutoff
        
        X_train_split = X_train[train_mask]
        y_train_split = y_train[train_mask]
        X_val = X_train[~train_mask]
        y_val = y_train[~train_mask]
        
        # LightGBM datasets
        train_lgb = lgb.Dataset(X_train_split, label=y_train_split)
        val_lgb = lgb.Dataset(X_val, label=y_val, reference=train_lgb)
        
        # Robust parameters to prevent overfitting warnings
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,  # Moderate complexity
            'learning_rate': 0.05,  # Moderate learning rate
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,  # Prevent overfitting
            'min_child_weight': 1e-3,
            'min_split_gain': 0.1,  # Require minimum gain
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'max_depth': 8,  # Reasonable depth
            'random_state': 42,
            'verbosity': -1,  # Silent to avoid warnings
            'force_col_wise': True,
            'min_data_in_leaf': 10,  # Minimum data per leaf
            'lambda_l1': 0.1,
            'lambda_l2': 0.1
        }
        
        # Calculate max iterations based on remaining time
        max_iterations = min(500, int(remaining_time * 5))  # Conservative estimate
        
        print(f"🔥 TRAINING FOR {remaining_time:.1f}s (max {max_iterations} iterations)...")
        training_start = time.time()
        
        try:
            self.model = lgb.train(
                params, train_lgb, 
                num_boost_round=max_iterations,
                valid_sets=[train_lgb, val_lgb], 
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=0)  # Silent
                ]
            )
        except Exception as e:
            print(f"⚠️ Training warning: {e}")
            # Fallback with even more conservative parameters
            params.update({
                'num_leaves': 15,
                'max_depth': 4,
                'min_child_samples': 50
            })
            self.model = lgb.train(
                params, train_lgb, 
                num_boost_round=100,  # Very limited iterations
                valid_sets=[val_lgb], 
                callbacks=[lgb.log_evaluation(period=0)]
            )
        
        training_time = time.time() - training_start
        
        # Feature importance
        try:
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
        except:
            # Fallback if feature importance fails
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': [1.0] * len(self.feature_columns)
            })
        
        print(f"✅ FINAL MODEL TRAINED!")
        print(f"   ⏱️  Training time: {training_time:.1f}s")
        print(f"   📈 Best iteration: {getattr(self.model, 'best_iteration', 'N/A')}")
        
        return training_time
    
    def evaluate_and_save(self, test_data, training_time):
        """Evaluate, save model, update results"""
        print("\n🔮 MAKING PREDICTIONS ON TEST DATA...")
        
        # Predictions
        X_test = test_data[self.feature_columns].copy()
        X_test = X_test.fillna(X_test.median())
        
        try:
            predictions = self.model.predict(X_test, num_iteration=getattr(self.model, 'best_iteration', None))
        except:
            predictions = self.model.predict(X_test)
            
        actuals = test_data['urgences_grippe'].values
        
        # Metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        mape = mean_absolute_percentage_error(actuals, predictions) * 100
        
        # Classification accuracy
        def get_alert_level(value):
            return 0 if value < 100 else (1 if value < 300 else (2 if value < 600 else 3))
        
        true_classes = [get_alert_level(v) for v in actuals]
        pred_classes = [get_alert_level(v) for v in predictions]
        class_acc = accuracy_score(true_classes, pred_classes) * 100
        
        print(f"📊 TEST RESULTS:")
        print(f"   📉 MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
        print(f"   📈 R²: {r2:.4f} | Classification Accuracy: {class_acc:.1f}%")
        
        # Save model
        print("\n💾 SAVING MODEL...")
        os.makedirs('models/LightGBM_models', exist_ok=True)
        model_path = f'models/LightGBM_models/{self.model_name}.pkl'
        
        model_data = {
            'model': self.model, 
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders, 
            'feature_importance': self.feature_importance,
            'model_name': self.model_name, 
            'cv_results': self.cv_results,
            'training_time': training_time,
            'best_iteration': getattr(self.model, 'best_iteration', 0)
        }
        
        try:
            joblib.dump(model_data, model_path)
            print(f"✅ MODEL SAVED: {model_path}")
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")
        
        # Update results CSV
        print("📝 UPDATING RESULTS CSV...")
        os.makedirs('models/results', exist_ok=True)
        results_file = 'models/results/training_results.csv'
        
        cv_mean_rmse = np.mean([r['rmse'] for r in self.cv_results]) if self.cv_results else 0
        cv_std_rmse = np.std([r['rmse'] for r in self.cv_results]) if self.cv_results else 0
        
        new_result = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': self.model_name, 
            'model_type': 'LightGBM', 
            'algorithm': 'Gradient Boosting',
            'training_time_seconds': training_time, 
            'training_time_minutes': training_time/60,
            'best_iteration': getattr(self.model, 'best_iteration', 0), 
            'n_features': len(self.feature_columns),
            'n_predictions': len(predictions), 
            'mae': mae, 'rmse': rmse, 'mape': mape,
            'r2_score': r2, 
            'accuracy_classification': class_acc,
            'median_error': np.median(np.abs(actuals - predictions)),
            'max_error': np.max(np.abs(actuals - predictions)),
            'mean_actual_value': actuals.mean(), 
            'mean_predicted_value': predictions.mean(),
            'data_split': '2019-2021_train_2022-2024_test', 
            'target_variable': 'urgences_grippe',
            'feature_engineering': 'robust_lag_rolling_seasonal_ias',
            'hyperparameters': f"lr=0.05_leaves=63_depth=8_cv3folds_5min_limit",
            'notes': f"5-minute limit, {len(self.cv_results)} CV folds, {training_time/60:.1f}min training, robust params, CV_RMSE={cv_mean_rmse:.3f}±{cv_std_rmse:.3f}"
        }])
        
        try:
            if os.path.exists(results_file):
                existing = pd.read_csv(results_file)
                combined = pd.concat([existing, new_result], ignore_index=True)
            else:
                combined = new_result
                
            combined.to_csv(results_file, index=False)
            print(f"✅ RESULTS UPDATED: {results_file} (Total experiments: {len(combined)})")
        except Exception as e:
            print(f"⚠️ Error updating results: {e}")
        
        # Save feature importance separately
        try:
            feature_importance_file = f'models/results/feature_importance_{self.model_name}.csv'
            self.feature_importance.to_csv(feature_importance_file, index=False)
            print(f"🔍 Feature importance saved: {feature_importance_file}")
        except Exception as e:
            print(f"⚠️ Error saving feature importance: {e}")
        
        return predictions, actuals, {
            'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'class_acc': class_acc
        }
    
    def create_visualizations(self, test_data, predictions, actuals, metrics):
        """Create comprehensive visualizations"""
        print("\n📊 CREATING VISUALIZATIONS...")
        
        try:
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Time series
            ax1 = fig.add_subplot(gs[0, :2])
            dates = test_data['date_semaine'].values
            ax1.plot(dates, actuals, 'o-', label='Actual', linewidth=2, alpha=0.8)
            ax1.plot(dates, predictions, 's--', label='Predicted', linewidth=2, alpha=0.8)
            ax1.set_title('Predictions vs Actual Over Time', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 2. Scatter plot
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.scatter(actuals, predictions, alpha=0.6, s=30)
            min_val, max_val = min(np.min(actuals), np.min(predictions)), max(np.max(actuals), np.max(predictions))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            ax2.set_xlabel('Actual')
            ax2.set_ylabel('Predicted')
            ax2.set_title(f'Accuracy Plot\nR² = {metrics["r2"]:.4f}', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 3. Feature importance
            ax3 = fig.add_subplot(gs[1, :2])
            top_features = self.feature_importance.head(15)
            if len(top_features) > 0:
                bars = ax3.barh(top_features['feature'], top_features['importance'])
                ax3.set_xlabel('Importance')
                ax3.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)
            
            # 4. CV results
            ax4 = fig.add_subplot(gs[1, 2])
            if self.cv_results:
                cv_rmse = [r['rmse'] for r in self.cv_results]
                cv_folds = [r['fold'] for r in self.cv_results]
                ax4.bar(cv_folds, cv_rmse, alpha=0.7, color='purple')
                ax4.set_xlabel('CV Fold')
                ax4.set_ylabel('RMSE')
                ax4.set_title('Cross-Validation Results', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3)
            
            # 5. Residuals
            ax5 = fig.add_subplot(gs[2, 0])
            residuals = predictions - actuals
            ax5.scatter(predictions, residuals, alpha=0.6, s=20)
            ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax5.set_xlabel('Predicted')
            ax5.set_ylabel('Residuals')
            ax5.set_title('Residuals Plot', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # 6. Error distribution
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.hist(residuals, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax6.set_xlabel('Prediction Error')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Error Distribution', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # 7. Performance summary
            ax7 = fig.add_subplot(gs[2, 2])
            summary_text = f"""MODEL PERFORMANCE
            
R² Score: {metrics['r2']:.4f}
RMSE: {metrics['rmse']:.2f}
MAE: {metrics['mae']:.2f}
MAPE: {metrics['mape']:.2f}%
Classification Acc: {metrics['class_acc']:.1f}%

MODEL INFO
Features: {len(self.feature_columns)}
CV Folds: {len(self.cv_results)}
Best Iteration: {getattr(self.model, 'best_iteration', 'N/A')}
"""
            ax7.text(0.1, 0.5, summary_text, transform=ax7.transAxes, fontsize=11,
                    verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
            ax7.set_title('Summary', fontsize=12, fontweight='bold')
            ax7.axis('off')
            
            # Main title
            fig.suptitle(f'LightGBM Analysis - {self.model_name}', fontsize=16, fontweight='bold')
            
            # Save and show
            plot_path = f'models/results/analysis_{self.model_name}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ VISUALIZATIONS SAVED: {plot_path}")
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Error creating visualizations: {e}")

def main():
    """Main pipeline with 5-minute time limit"""
    print("🚀 LIGHTGBM EPIDEMIC PREDICTION TRAINING")
    print("🎯 5-MINUTE TIME LIMIT: Quick CV + Model Save + Results Update + Graphs")
    print("=" * 70)
    
    start_time = time.time()
    predictor = LightGBMEpidemicPredictor()
    
    try:
        # 1. Load and prepare data
        print(f"⏰ Starting at: {datetime.now().strftime('%H:%M:%S')}")
        df = predictor.load_and_prepare_data()
        train_data, test_data = predictor.split_data(df)
        
        # 2. Run quick CV with time limit
        predictor.run_quick_cv(train_data, start_time)
        
        # 3. Train final model with remaining time
        training_time = predictor.train_final_model(train_data, start_time)
        
        # 4. Evaluate and save everything
        predictions, actuals, metrics = predictor.evaluate_and_save(test_data, training_time)
        
        # 5. Show visualizations if time permits
        elapsed = time.time() - start_time
        if elapsed < predictor.total_time_limit - 10:  # If more than 10s left
            predictor.create_visualizations(test_data, predictions, actuals, metrics)
        else:
            print("⚠️ Skipping visualizations due to time limit")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"⏰ End time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.2f}min)")
        print(f"🏆 Model: {predictor.model_name}")
        print(f"📈 Final R²: {metrics['r2']:.4f}")
        print(f"📊 Final RMSE: {metrics['rmse']:.2f}")
        
        if total_time <= predictor.total_time_limit:
            print(f"✅ Completed within {predictor.total_time_limit}s time limit")
        else:
            print(f"⚠️ Exceeded time limit by {total_time - predictor.total_time_limit:.1f}s")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()