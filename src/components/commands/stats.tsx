import type { ComponentCommand } from '@commands';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';
import { useEffect, useState } from 'preact/hooks';

const messages = i18n('stats', {
	commits: 'Commits (Last Year)',
	error: 'Unable to load GitHub statistics',
	followers: 'Followers',
	following: 'Following',
	loading: 'Loading GitHub stats...',
	profile: 'View GitHub Profile',
	recentActivity: 'Recent Activity',
	repositories: 'Repositories',
	stars: 'Stars',
	title: 'GitHub Statistics',
	topRepos: 'Top Repositories',
});

// Mock GitHub stats (in a real implementation, you'd fetch from GitHub API)
const mockStats = {
	commits: 127,
	followers: 23,
	following: 15,
	recentActivity: [
		{ message: 'Update cyber portal features', repo: 'cyber_portal', time: '2 hours ago', type: 'Push' },
		{ message: 'Created new AI project', repo: 'new-project', time: '1 day ago', type: 'Create' },
		{ message: 'Fix weather API integration', repo: 'WEATHER-APP', time: '3 days ago', type: 'Push' },
		{ message: 'Starred awesome AI repository', repo: 'awesome-ai', time: '1 week ago', type: 'Star' },
	],
	repositories: 38,
	stars: 1,
	topRepos: [
		{ description: 'Cyber Portal - Police Headquarters', language: 'HTML', name: 'cyber_portal', stars: 1 },
		{ description: 'Face mask detection using AI', language: 'Python', name: 'Face-Mask-Detector', stars: 0 },
		{ description: 'Flutter-Python weather app', language: 'Python', name: 'WEATHER-APP', stars: 1 },
		{ description: 'Personal portfolio repository', language: 'Markdown', name: 'vikasharma005', stars: 0 },
	],
};

const Stats: FunctionalComponent = () => {
	const t = useStore(messages);
	const [stats, setStats] = useState(mockStats);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(false);

	useEffect(() => {
		// Simulate API call
		const timer = setTimeout(() => {
			setLoading(false);
		}, 1000);

		return () => clearTimeout(timer);
	}, []);

	if (loading) {
		return (
			<div className='terminal-line-history'>
				<h3>{t.title}</h3>
				<p style={{ color: 'var(--color-text-200)', fontStyle: 'italic' }}>{t.loading}</p>
			</div>
		);
	}

	if (error) {
		return (
			<div className='terminal-line-history'>
				<h3>{t.title}</h3>
				<p style={{ color: 'var(--color-error)', fontStyle: 'italic' }}>{t.error}</p>
			</div>
		);
	}

	return (
		<div className='terminal-line-history'>
			<div
				style={{
					alignItems: 'center',
					display: 'flex',
					justifyContent: 'space-between',
					marginBottom: '1.5rem',
				}}>
				<h3>{t.title}</h3>
				<a
					href='https://github.com/vikasharma005'
					onMouseEnter={e => {
						e.currentTarget.style.background = 'var(--color-primary-hover)';
					}}
					onMouseLeave={e => {
						e.currentTarget.style.background = 'var(--color-primary)';
					}}
					rel='noopener noreferrer'
					style={{
						background: 'var(--color-primary)',
						borderRadius: '4px',
						color: 'white',
						fontSize: '0.9rem',
						padding: '0.5rem 1rem',
						textDecoration: 'none',
						transition: 'background 0.3s ease',
					}}
					target='_blank'>
					{t.profile}
				</a>
			</div>

			{/* Stats Grid */}
			<div
				style={{
					display: 'grid',
					gap: '1rem',
					gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
					marginBottom: '2rem',
				}}>
				<div
					style={{
						background: 'var(--color-bg-100)',
						border: '1px solid var(--color-border)',
						borderRadius: '8px',
						padding: '1rem',
						textAlign: 'center',
					}}>
					<div style={{ color: 'var(--color-primary)', fontSize: '2rem', fontWeight: 'bold' }}>
						{stats.repositories}
					</div>
					<div style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>{t.repositories}</div>
				</div>

				<div
					style={{
						background: 'var(--color-bg-100)',
						border: '1px solid var(--color-border)',
						borderRadius: '8px',
						padding: '1rem',
						textAlign: 'center',
					}}>
					<div style={{ color: 'var(--color-primary)', fontSize: '2rem', fontWeight: 'bold' }}>
						{stats.followers}
					</div>
					<div style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>{t.followers}</div>
				</div>

				<div
					style={{
						background: 'var(--color-bg-100)',
						border: '1px solid var(--color-border)',
						borderRadius: '8px',
						padding: '1rem',
						textAlign: 'center',
					}}>
					<div style={{ color: 'var(--color-primary)', fontSize: '2rem', fontWeight: 'bold' }}>
						{stats.following}
					</div>
					<div style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>{t.following}</div>
				</div>

				<div
					style={{
						background: 'var(--color-bg-100)',
						border: '1px solid var(--color-border)',
						borderRadius: '8px',
						padding: '1rem',
						textAlign: 'center',
					}}>
					<div style={{ color: 'var(--color-primary)', fontSize: '2rem', fontWeight: 'bold' }}>
						{stats.stars}
					</div>
					<div style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>{t.stars}</div>
				</div>

				<div
					style={{
						background: 'var(--color-bg-100)',
						border: '1px solid var(--color-border)',
						borderRadius: '8px',
						padding: '1rem',
						textAlign: 'center',
					}}>
					<div style={{ color: 'var(--color-primary)', fontSize: '2rem', fontWeight: 'bold' }}>
						{stats.commits}
					</div>
					<div style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>{t.commits}</div>
				</div>
			</div>

			{/* Top Repositories */}
			<div style={{ marginBottom: '2rem' }}>
				<h4 style={{ color: 'var(--color-primary)', marginBottom: '1rem' }}>{t.topRepos}</h4>
				<div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
					{stats.topRepos.map((repo, index) => (
						<div
							key={index}
							style={{
								alignItems: 'center',
								background: 'var(--color-bg-100)',
								border: '1px solid var(--color-border)',
								borderRadius: '6px',
								display: 'flex',
								justifyContent: 'space-between',
								padding: '0.75rem',
							}}>
							<div>
								<div style={{ color: 'var(--color-text)', fontWeight: 'bold' }}>{repo.name}</div>
								<div style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>
									{repo.description}
								</div>
							</div>
							<div style={{ alignItems: 'center', display: 'flex', gap: '0.5rem' }}>
								<span
									style={{
										background: 'var(--color-primary)',
										borderRadius: '12px',
										color: 'white',
										fontSize: '0.8rem',
										padding: '0.25rem 0.5rem',
									}}>
									{repo.language}
								</span>
								<span style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>
									⭐ {repo.stars}
								</span>
							</div>
						</div>
					))}
				</div>
			</div>

			{/* Recent Activity */}
			<div>
				<h4 style={{ color: 'var(--color-primary)', marginBottom: '1rem' }}>{t.recentActivity}</h4>
				<div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
					{stats.recentActivity.map((activity, index) => (
						<div
							key={index}
							style={{
								alignItems: 'center',
								background: 'var(--color-bg-100)',
								border: '1px solid var(--color-border)',
								borderRadius: '6px',
								display: 'flex',
								justifyContent: 'space-between',
								padding: '0.75rem',
							}}>
							<div>
								<div style={{ color: 'var(--color-text)', fontWeight: 'bold' }}>
									{activity.type} in {activity.repo}
								</div>
								<div style={{ color: 'var(--color-text-200)', fontSize: '0.9rem' }}>
									{activity.message}
								</div>
							</div>
							<div style={{ color: 'var(--color-text-200)', fontSize: '0.8rem' }}>{activity.time}</div>
						</div>
					))}
				</div>
			</div>
		</div>
	);
};

const StatsCommand: ComponentCommand = {
	command: 'stats',
	component: Stats,
};

export default StatsCommand;
