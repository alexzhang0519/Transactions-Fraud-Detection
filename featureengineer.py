import pandas as pd
from typing import Iterable, Optional, List

class FraudFeatureEngineer:
    def __init__(
        self,
        target_col: str = "target",
        use_chip_col: str = "use_chip",
        card_brand_col: str = "card_brand",
        card_type_col: str = "card_type",
        gender_col: str = "gender",
        has_chip_col: str = "has_chip",
        new_merchant_col: str = "is_new_merchant_for_client",
        card_on_dark_web_col: str = "card_on_dark_web",
        zip_col: str = "zip",
        current_age_col: str = "current_age",
        retirement_age_col: str = "retirement_age",
    ):
        self.target_col = target_col
        self.use_chip_col = use_chip_col
        self.card_brand_col = card_brand_col
        self.card_type_col = card_type_col
        self.gender_col = gender_col
        self.has_chip_col = has_chip_col
        self.new_merchant_col = new_merchant_col
        self.card_on_dark_web_col = card_on_dark_web_col
        self.zip_col = zip_col
        self.current_age_col = current_age_col
        self.retirement_age_col = retirement_age_col

    def add_is_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if self.target_col not in out.columns:
            raise KeyError(
                f"Column '{self.target_col}' does not exist."
            )
        out["is_fraud"] = (
            out[self.target_col]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq("yes")
        )
        return out

    def use_chip_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.use_chip_col not in out.columns:
            return out
        s = (
            out[self.use_chip_col]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        out["is_swipe_txn"] = (
            s.eq("swipe transaction").astype("int8")
        )
        out["is_chip_txn"] = (
            s.eq("chip transaction").astype("int8")
        )
        out["is_online_txn"] = (
            s.eq("online transaction").astype("int8")
        )
        out = out.drop(
            columns=[self.use_chip_col],
            errors="ignore"
        )
        return out
    
    def card_brand_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.card_brand_col not in out.columns:
            return out
        s = out[self.card_brand_col].astype("string").str.strip().str.lower()
        out["is_visa"] = (s == "visa").astype("int8")
        out["is_mastercard"] = (s == "mastercard").astype("int8")
        out["is_amex"] = (s == "amex").astype("int8")
        out["is_discover"] = (s == "discover").astype("int8")
        out = out.drop(columns=[self.card_brand_col], errors="ignore")
        return out

    def card_type_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.card_type_col not in out.columns:
            return out
        s = out[self.card_type_col].astype("string").str.strip().str.lower()
        out["is_credit_card"] = (s == "credit").astype("int8")
        out["is_debit_card"] = (s == "debit").astype("int8")
        out["is_prepaid_card"] = (s == "debit (prepaid)").astype("int8")
        out = out.drop(columns=[self.card_type_col], errors="ignore")
        return out
    
    def gender_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.gender_col not in out.columns:
            return out
        s = out[self.gender_col].astype("string").str.strip().str.lower()
        out["is_male"] = (s == "male").astype("int8")
        out = out.drop(columns=[self.gender_col], errors="ignore")
        return out
    
    def has_chip_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.has_chip_col not in out.columns:
            return out
        s = out[self.has_chip_col].astype("string").str.strip().str.lower()
        out["has_chip_dummy"] = (s == "yes").astype("int8")
        out = out.drop(columns=[self.has_chip_col], errors="ignore")
        return out

    def new_merchant_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.new_merchant_col not in out.columns:
            return out
        s = out[self.new_merchant_col].astype("string").str.strip().str.lower()
        out["new_merchant"] = (s == "true").astype("int8")
        out = out.drop(columns=[self.new_merchant_col], errors="ignore")
        return out
    
    def card_on_dark_web_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.card_on_dark_web_col not in out.columns:
            return out
        s = out[self.card_on_dark_web_col].astype("string").str.strip().str.lower()
        out["card_on_dark_web_dummy"] = (s == "yes").astype("int8")
        out = out.drop(columns=[self.card_on_dark_web_col], errors="ignore")
        return out

    def fix_zip_make_numeric(
        self,
        df: pd.DataFrame,
        out_col: str = "zip3_int",
        drop_zip: bool = True,
        zfill: int = 5,
        fillna_value=None,      #  -1 if you want no NaNs
        dtype: str = "float32", # float allows NaN
    ) -> pd.DataFrame:
        out = df.copy()
        if self.zip_col not in out.columns:
            return out
        z = out[self.zip_col].astype("string").str.strip()
        # keep only digits; non-digits -> NA
        z = z.where(z.str.fullmatch(r"\d+"), pd.NA).str.zfill(zfill)
        zip3 = z.str.slice(0, 3)
        zip3_num = pd.to_numeric(zip3, errors="coerce")  # float with NaN
        if fillna_value is not None:
            zip3_num = zip3_num.fillna(fillna_value)
        out[out_col] = zip3_num.astype(dtype)
        if drop_zip:
            out = out.drop(columns=[self.zip_col], errors="ignore")
        return out
    
    def add_years_to_retirement(
        self,
        df: pd.DataFrame,
        out_col: str = "years_to_retirement",
        min_years: float = -40,
        max_years: float = 60,
        drop_retirement_age: bool = True,
        dtype: str = "float32",
    ) -> pd.DataFrame:
        out = df.copy()

        if {self.current_age_col, self.retirement_age_col}.issubset(out.columns):
            ytr = out[self.retirement_age_col] - out[self.current_age_col]
            ytr = ytr.where(ytr.between(min_years, max_years))
            out[out_col] = ytr.astype(dtype)

        if drop_retirement_age:
            out = out.drop(columns=[self.retirement_age_col], errors="ignore")

        return out
    
    def periodM_to_month_index(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in cols:
            s = out[col]
            out[col] = (s.dt.year.astype("int32") * 12 + s.dt.month.astype("int16")).astype("Int32")
        return out
    
    def drop_columns_for_model(
        self,
        df: pd.DataFrame,
        drop_cols: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        out = df.copy()
        DEFAULT_DROP_COLS: List[str] = [
            "id", "fraud_id", "client_id", "id_user", "merchant_id",
            "card_id", "id_card", "card_number",
            "date", "merchant_city", "merchant_state", "mcc_description", "address",
            "birth_date", "age_group", self.target_col, "cvv",
        ]
        cols = list(drop_cols) if drop_cols is not None else DEFAULT_DROP_COLS
        return out.drop(columns=[c for c in cols if c in out.columns], errors="ignore")
    
    def dropna_target(self, df: pd.DataFrame, target_col: str = "is_fraud") -> pd.DataFrame:
        before = len(df) #count # of rows
        out = df.dropna(subset=[target_col]).copy()
        after = len(out)
        print(f"[dropna_target] Dropped {before-after:,} rows (kept {after:,} / {before:,}).")
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = self.add_is_fraud(out)
        out = self.use_chip_dummies(out)
        out = self.card_brand_dummies(out)
        out = self.card_type_dummies(out)
        out = self.gender_dummies(out)
        out = self.has_chip_dummies(out)
        out = self.new_merchant_dummies(out)
        out = self.card_on_dark_web_dummies(out)
        out = self.fix_zip_make_numeric(out)
        out = self.add_years_to_retirement(out)
        out = self.periodM_to_month_index(out, ["expires","acct_open_date"]) 
        out = self.drop_columns_for_model(out)
        out = self.dropna_target(out)
        return out