import type { ComponentCommand } from '@commands';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';

const messages = i18n('achievements', {
	title: 'Achievements and Recognition',
	award1: 'UDHBHAV-2024 Software Project Winner',
	award1Date: 'Nov 2024',
	award1Org: 'Poornima Institute of Engineering & Technology',
	award1Desc: '1st place in Software category at UDHBHAV-2024 National Level Project Exhibition. Recognized for innovative and highly functional software project demonstrating outstanding technical skills and creative problem-solving.',
	award2: 'Outstanding Contribution Award',
	award2Date: 'Oct 2024',
	award2Org: 'Poornima Institute of Engineering & Technology',
	award2Desc: 'Presented during Kalanidhi Annual Award Ceremony for significant contributions to academic excellence, leadership, and project development. Recognized for spearheading high-impact projects and fostering innovation.',
	award3: 'UDHBHAV-2023 Software Project Runner Up',
	award3Date: 'Sep 2023',
	award3Org: 'Poornima Institute of Engineering & Technology',
	award3Desc: '2nd position in Software category at UDHBHAV-2023 National Level Project Exhibition. Recognized by panel of industry experts and academics for strong technical expertise and innovation.',
});

const Achievements: FunctionalComponent = () => {
	const t = useStore(messages);

	return (
		<div className='terminal-line-history'>
			<h3>{t.title}</h3>
			
			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.award1}</strong>
				</p>
				<p>
					{t.award1Org} | {t.award1Date}
				</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.award1Desc}</p>
			</div>

			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.award2}</strong>
				</p>
				<p>
					{t.award2Org} | {t.award2Date}
				</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.award2Desc}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.award3}</strong>
				</p>
				<p>
					{t.award3Org} | {t.award3Date}
				</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.award3Desc}</p>
			</div>
		</div>
	);
};

const AchievementsCommand: ComponentCommand = {
	command: 'achievements',
	component: Achievements,
};

export default AchievementsCommand;
