"""
Study Service - Analytics and data analysis for inference runs.

This service provides business logic for analyzing stored inference data,
transforming raw database queries into analysis-ready DataFrames.
"""

import pandas as pd
from typing import Optional
from datetime import datetime, timedelta, timezone
from ..db.inference_repository import InferenceRepository


class StudyService:
    """
    Service for analyzing and studying stored inference data.
    
    This service provides high-level analytics methods that transform
    raw database queries into pandas DataFrames, making it easy to
    visualize and analyze inference patterns.
    
    Usage:
        with db.session_scope() as session:
            repo = InferenceRepository(session)
            study = StudyService(repo)
            
            # Get provider comparison
            comparison = study.get_provider_comparison()
            
            # Get recent inferences
            recent = study.get_recent_inferences(limit=10)
    """

    def __init__(self, repository: InferenceRepository):
        """
        Initialize the study service.
        
        Args:
            repository: InferenceRepository instance for database access
        """
        self.repository = repository

    def get_provider_comparison(self) -> pd.DataFrame:
        """
        Compare providers by various metrics.
        
        Returns a DataFrame with provider statistics including:
        - Count of inferences per provider
        - Average tokens per provider
        - Average latency per provider
        - Total tokens per provider
        - Estimated cost (derived metric)
        
        Returns:
            DataFrame with provider comparison metrics.
            Empty DataFrame if no data exists.
        """
        stats = self.repository.get_provider_stats()
        
        if not stats:
            return pd.DataFrame()
        
        df = pd.DataFrame(stats)

        if not df.empty:
            # Add derived metrics
            # Example pricing: $0.00002 per token (adjust based on actual pricing)
            df['avg_cost_estimate'] = df['avg_tokens'] * 0.00002
            df['total_cost_estimate'] = df['total_tokens'] * 0.00002

        return df

    def get_recent_inferences(
        self,
        limit: int = 50,
        provider: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get recent inference runs as a DataFrame.
        
        Args:
            limit: Maximum number of results (default: 50)
            provider: Optional provider filter (default: None)
        
        Returns:
            DataFrame with recent inferences, including:
            - id, prompt (truncated), response (truncated), provider,
              tokens, latency_ms, created_at
        """
        runs = self.repository.query_inferences(
            provider=provider,
            limit=limit
        )

        if not runs:
            return pd.DataFrame()

        data = [
            {
                'id': run.id,
                'prompt': run.prompt[:100] + '...' if len(run.prompt) > 100 else run.prompt,
                'response': run.response[:100] + '...' if len(run.response) > 100 else run.response,
                'provider': run.provider,
                'tokens': run.tokens,
                'latency_ms': run.latency_ms,
                'created_at': run.created_at
            }
            for run in runs
        ]

        df = pd.DataFrame(data)
        
        # Convert created_at to datetime if not already
        if not df.empty and 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])

        return df

    def get_time_series_data(
        self,
        days: int = 7,
        group_by: str = 'day'
    ) -> pd.DataFrame:
        """
        Get time-series data for visualization.
        
        Aggregates inference data by time period and provider, useful
        for creating line charts showing trends over time.
        
        Args:
            days: Number of days to look back (default: 7)
            group_by: Time period to group by - 'day', 'hour', or 'minute' (default: 'day')
        
        Returns:
            DataFrame with time-series data grouped by time period and provider.
            Columns include: created_at, provider, tokens (sum/mean), latency_ms (mean)
            Empty DataFrame if no data exists.
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        runs = self.repository.query_inferences(
            date_range=(start_date, end_date),
            limit=10000  # Large limit for aggregation
        )

        if not runs:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                'created_at': run.created_at,
                'provider': run.provider,
                'tokens': run.tokens or 0,
                'latency_ms': run.latency_ms or 0.0
            }
            for run in runs
        ])

        if df.empty:
            return pd.DataFrame()

        df['created_at'] = pd.to_datetime(df['created_at'])

        # Group by time period
        freq_map = {'day': 'D', 'hour': 'h', 'minute': 'min'}
        freq = freq_map.get(group_by, 'D')

        grouped = df.groupby([pd.Grouper(key='created_at', freq=freq), 'provider']).agg({
            'tokens': ['sum', 'mean'],
            'latency_ms': 'mean'
        }).reset_index()

        # Flatten column names
        grouped.columns = ['created_at', 'provider', 'tokens_sum', 'tokens_mean', 'latency_ms_mean']

        return grouped

    def search_prompts(self, search_term: str, limit: int = 100) -> pd.DataFrame:
        """
        Search historical prompts by keyword.
        
        Performs case-insensitive substring search on prompt content.
        
        Args:
            search_term: Text to search for in prompts
            limit: Maximum number of results (default: 100)
        
        Returns:
            DataFrame with matching prompts and responses.
            Columns include: id, prompt, response (truncated), provider, created_at
            Empty DataFrame if no matches found.
        """
        runs = self.repository.search_by_prompt(search_term, limit=limit)

        if not runs:
            return pd.DataFrame()

        data = [
            {
                'id': run.id,
                'prompt': run.prompt,
                'response': run.response[:200] + '...' if len(run.response) > 200 else run.response,
                'provider': run.provider,
                'created_at': run.created_at
            }
            for run in runs
        ]

        df = pd.DataFrame(data)
        
        # Convert created_at to datetime if not already
        if not df.empty and 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])

        return df

    def get_summary_stats(self) -> dict:
        """
        Get overall summary statistics.
        
        Returns high-level metrics useful for dashboard overviews.
        
        Returns:
            Dictionary with summary statistics:
            - total_inferences: Total number of inference runs
            - total_tokens: Sum of all tokens across all providers
            - unique_providers: Number of distinct providers used
            - provider_breakdown: List of provider stats (from get_provider_stats)
        """
        total = self.repository.get_total_count()
        provider_stats = self.repository.get_provider_stats()

        total_tokens = sum(s['total_tokens'] for s in provider_stats)

        return {
            'total_inferences': total,
            'total_tokens': total_tokens,
            'unique_providers': len(provider_stats),
            'provider_breakdown': provider_stats
        }

